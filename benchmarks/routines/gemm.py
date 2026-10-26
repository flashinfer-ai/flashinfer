from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from einops import einsum

import flashinfer
from flashinfer.autotuner import autotune
from flashinfer.fp8_quantization import mxfp8_quantize
from flashinfer.testing.utils import (
    bench_gpu_time,
    dequantize_fp8,
    quantize_fp8,
)

from .flashinfer_benchmark_utils import (
    dtype_str_to_torch_dtype,
    get_device,
    print_perf_metrics,
    is_close_stats,
    filter_backends_by_compute_capability,
    warn_if_pdl_unsupported,
)


def run_gemm_test(args):
    """
    Run a gemm test.

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.routine == "gemm_fp8_nt_groupwise":
        return testGemmFp8NtGroupwise(args)
    elif args.routine == "group_gemm_fp8_nt_groupwise":
        return testGroupGemmFp8NtGroupwise(args)
    elif args.routine == "bmm_fp8":
        return testBmmFp8(args)
    elif args.routine == "mm_fp8":
        return testMmFp8(args)
    elif args.routine == "bmm_mxfp8":
        return testBmmMxfp8(args)
    elif args.routine == "mm_fp4":
        return testMmFp4(args)
    elif args.routine == "mm_bf16_fp4":
        return testMmBf16Fp4(args)
    elif args.routine == "mm_mxfp8":
        return testMmMxfp8(args)
    elif args.routine == "mm_bf16":
        return testMmBf16(args)
    elif args.routine == "bmm_bf16":
        return testBmmBf16(args)
    elif args.routine == "tinygemm_bf16":
        return testTinygemmBf16(args)
    elif args.routine == "router_gemm":
        return testRouterGemm(args)
    elif args.routine == "group_gemm_mxfp4_nt_groupwise":
        return testGroupGemmMxfp4NtGroupwise(args)
    elif args.routine == "masked_grouped_gemm":
        return testMaskedGroupedGemm(args)
    elif args.routine == "grouped_mm":
        return testGroupedMm(args)
    elif args.routine == "mm_nvfp4_svdquant":
        return testMmNvfp4Svdquant(args)
    elif args.routine == "fused_linear":
        return testFusedLinear(args)
    else:
        raise ValueError(f"Unsupported routine: {args.routine}")


def parse_gemm_args(line, parser):
    """
    Parse command line arguments for gemm test configuration.

    Args:
        line: Command line arguments
        parser: ArgumentParser object already populated with shared arguments

    Returns:
        Parsed argument namespace
    """
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=1,
        help="Batch size of test case.",
    )
    parser.add_argument(
        "--m", type=int, required=True, help="Number of rows in the first matrix."
    )
    parser.add_argument(
        "--n", type=int, required=True, help="Number of columns in the second matrix."
    )
    parser.add_argument(
        "--k",
        type=int,
        required=True,
        help="Number of columns in the first matrix and number of rows in the second matrix.",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        required=False,
        default=128,
        help="Tile size for the gemm operation.",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        required=False,
        default=1,
        help="Group size for the group gemm operation.",
    )
    parser.add_argument(
        "--scale_major_mode",
        type=str,
        required=False,
        default="MN",
        choices=["MN", "K"],
        help="Scale major mode.",
    )
    parser.add_argument(
        "--input_dtype",
        type=str,
        required=False,
        default="fp8_e4m3",
        help="Data type of the input.",
    )
    parser.add_argument(
        "--mat2_dtype",
        type=str,
        required=False,
        default="fp8_e4m3",
        help="Data type of the mat2.",
    )
    parser.add_argument(
        "--out_dtype",
        type=str,
        required=False,
        default="bfloat16",
        help="Data type of the output.",
    )
    parser.add_argument(
        "--mma_sm",
        type=int,
        required=False,
        default=1,
        choices=[1, 2],
        help="How many SMs to use for the MMA operation, must be 1 or 2",
    )
    parser.add_argument(
        "--backends",
        type=str,
        required=False,
        nargs="+",
        default=["cudnn"],
        choices=[
            "cudnn",
            "cublas",
            "trtllm",
            "cutlass",
            "tgv",
            "cublaslt",
            "cute-dsl",
            "cute-dsl-native",
            "blackwell-native",
            "blackwell-tiled",
            "b12x",
            "auto",
            "tinygemm",
            "cutile",
            "trtllm_low_latency",
            "cutedsl_low_latency",
            "deepgemm",
            "dsv3",
            "cute-dsl-unfused",
            "cutlass-sm120",
            "prims-ts",
        ],
        help="Kernel backends to test. Default: cudnn",
    )
    parser.add_argument(
        "--use_128x4_sf_layout",
        action="store_true",
        help="Use 128x4 SF layout for the input and mat2.",
    )
    parser.add_argument(
        "--use_nvfp4",
        action="store_true",
        help="In mm_fp4, whether to use nvfp4 quantization or mxfp4 quantization, defaults to False.",
    )
    parser.add_argument(
        "--autotune",
        action="store_true",
        default=False,
        help=(
            "Enable autotuner warmup for supported routines (mm_fp4, bmm_fp8, mm_fp8, bmm_mxfp8, mm_mxfp8, mm_bf16, bmm_bf16)."
        ),
    )
    parser.add_argument(
        "--bias",
        action="store_true",
        default=False,
        help="Use bias (mm_bf16 TGV/TinyGEMM/cuDNN/CuTe-DSL, tinygemm_bf16, mm_nvfp4_svdquant, fused_linear)",
    )
    parser.add_argument(
        "--scale_granularity_m",
        type=int,
        required=False,
        default=1,
        help=(
            "gemm_fp8_nt_groupwise: rows covered by one A scale. 1 (default) is the "
            "1x128 groupwise path; 128 benchmarks gemm_fp8_nt_blockscaled (128x128 A scales)."
        ),
    )
    parser.add_argument(
        "--expected_m",
        type=int,
        required=False,
        default=None,
        help="masked_grouped_gemm: mean valid rows per group (default: m // 2).",
    )
    parser.add_argument(
        "--rank",
        type=int,
        required=False,
        default=32,
        help="mm_nvfp4_svdquant: LoRA rank r (positive multiple of 32).",
    )
    parser.add_argument(
        "--epilogue",
        type=str,
        required=False,
        default="linear",
        choices=["linear", "swiglu", "qkv_qknorm_rope"],
        help=(
            "fused_linear: fused epilogue. swiglu halves N (interleaved gate/up); "
            "qkv_qknorm_rope needs N % 384 == 0 (head_dim 128)."
        ),
    )

    args = parser.parse_args(line)
    has_backends_arg = any(
        token == "--backends" or token.startswith("--backends=") for token in line
    )
    has_input_dtype_arg = any(
        token == "--input_dtype" or token.startswith("--input_dtype=") for token in line
    )
    has_mat2_dtype_arg = any(
        token == "--mat2_dtype" or token.startswith("--mat2_dtype=") for token in line
    )
    if args.routine == "tinygemm_bf16":
        if not has_backends_arg:
            args.backends = ["tinygemm"]
        if not has_input_dtype_arg:
            args.input_dtype = "bfloat16"
        if not has_mat2_dtype_arg:
            args.mat2_dtype = "bfloat16"
    if args.routine == "mm_bf16_fp4":
        if not has_backends_arg:
            args.backends = ["cute-dsl"]
        if not has_input_dtype_arg:
            args.input_dtype = "bfloat16"
    if args.routine == "mm_fp8":
        if not has_backends_arg:
            args.backends = ["trtllm_low_latency"]
    # Defaults for routines whose library API has no cudnn backend. Existing
    # group_gemm_fp8_nt_groupwise lines never passed --backends (it was
    # hard-coded to cutlass), so keep cutlass as their default.
    default_backends = {
        "group_gemm_fp8_nt_groupwise": ["cutlass"],
        "router_gemm": ["dsv3"],
        "group_gemm_mxfp4_nt_groupwise": ["cutlass"],
        "mm_nvfp4_svdquant": ["cutlass"],
        "fused_linear": ["prims-ts"],
    }
    if args.routine in default_backends and not has_backends_arg:
        args.backends = list(default_backends[args.routine])
    if args.routine == "router_gemm" and not has_input_dtype_arg:
        args.input_dtype = "bfloat16"
    if args.verbose >= 1:
        print(f"[INFO] {args = }")
    return args


def to_float8(x, dtype=torch.float8_e4m3fn):
    finfo = torch.finfo(dtype)
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    scale = finfo.max / amax
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    return x_scl_sat.to(dtype), scale.float().reciprocal()


def _drop_failing_backends(backends, trial):
    """Probe-and-drop: call ``trial(backend)`` once per backend and remove the
    backends whose launch raises (shape, layout or dtype limits that a
    compute-capability whitelist cannot express). Mutates and returns
    ``backends``."""
    for backend in list(backends):
        try:
            trial(backend)
        except Exception as e:
            print(
                f"[INFO] {backend} backend does not support this configuration: {type(e).__name__}: {e}"
            )
            backends.remove(backend)
    return backends


def testGemmFp8NtGroupwise(args):
    """
    Test gemm_fp8_nt_groupwise API.

    This test:
    1. Generates random input tensors
    2. Quantizes input tensors to FP8
    3. Runs gemm_fp8_nt_groupwise
    4. Runs reference check
    5. Measures performance metrics (TFLOPS, TB/sec)

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    warn_if_pdl_unsupported(args, args.routine)
    if args.verbose >= 1:
        print("[INFO] Running testGemmFp8NtGroupwise")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    ## Parse input arguments
    backends = args.backends
    m = args.m
    n = args.n
    k = args.k
    tile_size = args.tile_size
    scale_major_mode = args.scale_major_mode
    mma_sm = args.mma_sm
    is_cuda_graph_compatible = not args.no_cuda_graph
    run_refcheck = args.refcheck
    res = []

    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    out_dtype = dtype_str_to_torch_dtype(args.out_dtype)
    if out_dtype not in [torch.bfloat16, torch.float16]:
        raise ValueError(f"Unsupported output dtype: {args.out_dtype}")
    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)
    if input_dtype not in [torch.float8_e4m3fn, torch.float8_e5m2]:
        raise ValueError(f"Unsupported input dtype: {args.input_dtype}")
    if dtype_str_to_torch_dtype(args.mat2_dtype) != input_dtype:
        raise ValueError("gemm_fp8_nt_groupwise requires --mat2_dtype == --input_dtype")
    # --scale_granularity_m 128 benchmarks gemm_fp8_nt_blockscaled: A is scaled
    # per 128x128 tile instead of per 1x128 row group.
    scale_granularity_m = args.scale_granularity_m
    if scale_granularity_m not in [1, tile_size]:
        raise ValueError(
            f"--scale_granularity_m must be 1 or --tile_size ({tile_size}), got {scale_granularity_m}"
        )
    ## Done parsing input arguments

    # Only CUTLASS implements e5m2 and the 128x128 A-scale (blockscaled) wrapper;
    # the other kernels are e4m3 with 1x128 A scales.
    if input_dtype != torch.float8_e4m3fn or scale_granularity_m != 1:
        for backend in [b for b in backends if b != "cutlass"]:
            print(
                f"[INFO] {backend} only supports fp8_e4m3 with 1x{tile_size} A scales, removing {backend} from backends"
            )
            backends.remove(backend)

    if "trtllm" in backends:
        remove_trtllm = False
        if scale_major_mode != "MN":
            print(
                "[INFO] trtllm only supports MN scale_major_mode, removing trtllm from backends"
            )
            remove_trtllm = True
        if k < 256 or k % 256 != 0:
            # The API only checks k >= 256; k % 256 == 128 aborts inside the
            # kernel runner ("unsupported tactic id") instead of raising.
            print(
                "[INFO] trtllm only supports k >= 256 and k % 256 == 0, removing trtllm from backends"
            )
            remove_trtllm = True
        if out_dtype != torch.bfloat16:
            print("[INFO] trtllm only writes bfloat16, removing trtllm from backends")
            remove_trtllm = True
        if remove_trtllm:
            backends.remove("trtllm")

    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    ## Prepare input tensors
    a_val = torch.randn((m, k), dtype=torch.float, device=device)
    b_val = torch.randn((n, k), dtype=torch.float, device=device) / np.sqrt(k)

    if args.verbose >= 2:
        print(f"[VVERBOSE] {a_val.shape = }")
        print(f"[VVERBOSE] {b_val.shape = }")

    if scale_major_mode == "K":
        a_scale_shape = (m // scale_granularity_m, k // tile_size)
        b_scale_shape = (n // tile_size, k // tile_size)
    else:
        a_scale_shape = (k // tile_size, m // scale_granularity_m)
        b_scale_shape = (k // tile_size, n // tile_size)

    a_tile_shape = (scale_granularity_m, tile_size)
    b_tile_shape = (tile_size, tile_size)

    a_fp8, a_scale = quantize_fp8(a_val, a_scale_shape, a_tile_shape, scale_major_mode)
    b_fp8, b_scale = quantize_fp8(b_val, b_scale_shape, b_tile_shape, scale_major_mode)
    if input_dtype == torch.float8_e5m2:
        # quantize_fp8 emits e4m3; re-encode as e5m2 under the same scales so
        # the dequantized reference below is exact for the e5m2 values.
        a_fp8 = a_fp8.float().to(torch.float8_e5m2)
        b_fp8 = b_fp8.float().to(torch.float8_e5m2)

    if args.verbose >= 2:
        print(f"[VVERBOSE] {a_fp8.shape = }")
        print(f"[VVERBOSE] {b_fp8.shape = }")
        print(f"[VVERBOSE] {a_scale.shape = }")
        print(f"[VVERBOSE] {b_scale.shape = }")

    a_dequant = dequantize_fp8(a_fp8, a_scale, scale_major_mode)
    b_dequant = dequantize_fp8(b_fp8, b_scale, scale_major_mode)
    # trtllm consumes the MN-major b_scale transposed, (n // 128, k // 128),
    # as tests/gemm/test_groupwise_scaled_gemm_fp8.py does. Without it most
    # outputs are wrong.
    b_scale_trtllm = b_scale.t().contiguous() if "trtllm" in backends else None

    def run_backend(backend, a_fp8, b_fp8, a_scale, b_scale):
        if backend == "cutlass" and scale_granularity_m != 1:
            return flashinfer.gemm.gemm_fp8_nt_blockscaled(
                a_fp8,
                b_fp8,
                a_scale,
                b_scale,
                scale_major_mode=scale_major_mode,
                mma_sm=mma_sm,
                out_dtype=out_dtype,
            )
        if backend in ["cutlass", "trtllm", "cutile", "cute-dsl", "auto"]:
            return flashinfer.gemm.gemm_fp8_nt_groupwise(
                a=a_fp8,
                b=b_fp8,
                a_scale=a_scale,
                b_scale=b_scale_trtllm if backend == "trtllm" else b_scale,
                scale_major_mode=scale_major_mode,
                out_dtype=out_dtype,
                mma_sm=mma_sm,
                backend=backend,
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    # The cc whitelist is coarse; drop backends whose guard rejects this
    # problem (cutile: K-major only, cute-dsl: SM107 tiling limits, ...)
    # instead of failing the whole case in the timed loop.
    backends = _drop_failing_backends(
        backends, lambda bk: run_backend(bk, a_fp8, b_fp8, a_scale, b_scale)
    )
    if len(backends) == 0:
        print("[ERROR] No backends passed validation. Exiting.")
        return res

    has_reference_output = False
    if run_refcheck:
        reference_output = einsum(a_dequant, b_dequant, "m k, n k -> m n").to(out_dtype)
        has_reference_output = True

    # Storage for timing results and outputs
    backend_times = {backend: [] for backend in backends}
    outputs = {}
    for cur_backend in backends:
        if run_refcheck:
            outputs[cur_backend] = run_backend(
                cur_backend, a_fp8, b_fp8, a_scale, b_scale
            ).detach()
        backend_times[cur_backend] = bench_gpu_time(
            fn=run_backend,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            sleep_after_run=True,  # GEMMs are very MMA-heavy, so prefer sleep to reduce throttling.
            enable_cupti=args.use_cupti,
            use_cuda_graph=is_cuda_graph_compatible,
            cold_l2_cache=True,
            input_args=(cur_backend, a_fp8, b_fp8, a_scale, b_scale),
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
                        f"[ERROR] Output tensor mismatch from backend {tested_backends[i]}"
                    )
                    if not args.allow_output_mismatch:
                        raise AssertionError(
                            f"[ERROR] Backend {tested_backends[i]} output mismatch with {num_different_elements} elements"
                        )

    for backend in backends:
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])

            problem_flops = 2 * m * n * k
            problem_bytes = (m * k + n * k) * torch.float8_e4m3fn.itemsize + (
                m * n
            ) * out_dtype.itemsize
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
            cur_res["n"] = n
            cur_res["k"] = k
            cur_res["tile_size"] = tile_size
            cur_res["scale_major_mode"] = scale_major_mode
            # Recorded only off the default path, so existing rows keep their
            # problem key (these columns were blank for this routine before).
            if scale_granularity_m != 1:
                cur_res["scale_granularity_m"] = scale_granularity_m
            if input_dtype != torch.float8_e4m3fn:
                cur_res["input_dtype"] = input_dtype
            cur_res["out_dtype"] = out_dtype
            cur_res["mma_sm"] = mma_sm
            cur_res["backend"] = backend
            cur_res["case_tag"] = args.case_tag
            res.append(cur_res)
    return res


def testGroupGemmFp8NtGroupwise(args):
    """
    Test group_gemm_fp8_nt_groupwise API.

    This test:
    1. Generates random input tensors
    2. Quantizes input tensors to FP8
    3. Runs group_gemm_fp8_nt_groupwise
    4. Runs reference check
    5. Measures performance metrics (TFLOPS, TB/sec)

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    warn_if_pdl_unsupported(args, args.routine)
    if args.verbose >= 1:
        print("[INFO] Running testGroupGemmFp8NtGroupwise")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    ## Parse input arguments
    # Backends (CSV label -> library entry point):
    #   cutlass   group_gemm_fp8_nt_groupwise(backend="trtllm"): the API's
    #             default literal "trtllm" runs the CUTLASS sm100/sm120 kernel
    #   cutile    group_gemm_fp8_nt_groupwise(backend="cutile")      K-major only
    #   deepgemm  group_deepgemm_fp8_nt_groupwise (m_indices)         K-major only
    #   cute-dsl  group_gemm_fp8_nt_groupwise_contiguous (m_indices)  K-major only
    backends = list(args.backends)
    m = args.m
    n = args.n
    k = args.k
    group_size = args.group_size
    tile_size = args.tile_size
    scale_major_mode = args.scale_major_mode
    mma_sm = args.mma_sm
    is_cuda_graph_compatible = not args.no_cuda_graph
    run_refcheck = args.refcheck
    res = []

    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    for backend in ["cutile", "deepgemm", "cute-dsl"]:
        if backend in backends and scale_major_mode != "K":
            print(
                f"[INFO] {backend} only supports K scale_major_mode, removing {backend} from backends"
            )
            backends.remove(backend)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    out_dtype = dtype_str_to_torch_dtype(args.out_dtype)
    if out_dtype not in [torch.bfloat16, torch.float16]:
        raise ValueError(f"Unsupported output dtype: {args.out_dtype}")
    ## Done parsing input arguments
    ## Prepare input tensors
    a_val = torch.randn((group_size * m, k), dtype=torch.float, device="cuda")
    b_val = torch.randn((group_size, n, k), dtype=torch.float, device="cuda") / np.sqrt(
        k
    )

    if args.verbose >= 2:
        print(f"[VVERBOSE] {a_val.shape = }")
        print(f"[VVERBOSE] {b_val.shape = }")

    if scale_major_mode == "K":
        a_scale_shape = (group_size * m, k // tile_size)
        b_scale_shape = (group_size, n // tile_size, k // tile_size)
    else:
        a_scale_shape = (k // tile_size, m * group_size)
        b_scale_shape = (group_size, k // tile_size, n // tile_size)

    a_tile_shape = (1, tile_size)
    b_tile_shape = (1, tile_size, tile_size)

    a_fp8, a_scale = quantize_fp8(a_val, a_scale_shape, a_tile_shape, scale_major_mode)
    b_fp8, b_scale = quantize_fp8(b_val, b_scale_shape, b_tile_shape, scale_major_mode)

    a_dequant = dequantize_fp8(a_fp8, a_scale, scale_major_mode)
    b_dequant = dequantize_fp8(b_fp8, b_scale, scale_major_mode)

    m_indptr = torch.arange(0, group_size + 1, dtype=torch.int32, device="cuda") * m
    # deepgemm / cute-dsl take the contiguous layout: one expert index per row
    # of a instead of segment offsets.
    m_indices = torch.repeat_interleave(
        torch.arange(group_size, dtype=torch.int32, device="cuda"), m
    )
    # DeepGEMM repacks fp32 scales into its UE8M0 int layout on every call
    # (several small kernels plus host work); serving stacks pass pre-packed
    # scales, so pack them once here with the library's own transform.
    deepgemm_scales = None
    if "deepgemm" in backends:
        try:
            deepgemm_scales = _deepgemm_packed_scales(
                a_scale, b_scale, group_size * m, n, k, group_size
            )
        except Exception as e:
            print(
                f"[INFO] deepgemm backend does not support this configuration: {type(e).__name__}: {e}"
            )
            backends.remove("deepgemm")

    if args.verbose >= 2:
        print(f"[VVERBOSE] {a_fp8.shape = }")
        print(f"[VVERBOSE] {b_fp8.shape = }")
        print(f"[VVERBOSE] {a_scale.shape = }")
        print(f"[VVERBOSE] {b_scale.shape = }")
        print(f"[VVERBOSE] {m_indptr.shape = }")

    def run_backend(backend, a_fp8, b_fp8, a_scale, b_scale, m_indptr):
        if backend in ["cutlass", "cutile"]:
            return flashinfer.gemm.group_gemm_fp8_nt_groupwise(
                a=a_fp8,
                b=b_fp8,
                a_scale=a_scale,
                b_scale=b_scale,
                m_indptr=m_indptr,
                scale_major_mode=scale_major_mode,
                out_dtype=out_dtype,
                mma_sm=mma_sm,
                backend="cutile" if backend == "cutile" else "trtllm",
            )
        elif backend == "deepgemm":
            return flashinfer.gemm.group_deepgemm_fp8_nt_groupwise(
                a_fp8, b_fp8, *deepgemm_scales, m_indices, out_dtype=out_dtype
            )
        elif backend == "cute-dsl":
            return flashinfer.gemm.group_gemm_fp8_nt_groupwise_contiguous(
                a_fp8, b_fp8, a_scale, b_scale, m_indices, out_dtype=out_dtype
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    # Drop backends whose API rejects this problem (deepgemm / cute-dsl need
    # 128-row aligned groups, cute-dsl a bfloat16 output, ...).
    backends = _drop_failing_backends(
        backends,
        lambda bk: run_backend(bk, a_fp8, b_fp8, a_scale, b_scale, m_indptr),
    )
    if len(backends) == 0:
        print("[ERROR] No backends passed validation. Exiting.")
        return res

    has_reference_output = False
    if run_refcheck:
        reference_output = (
            einsum(
                a_dequant.view((group_size, m, k)), b_dequant, "b m k, b n k -> b m n"
            )
            .view((group_size * m, n))
            .to(out_dtype)
        )
        has_reference_output = True

    # Storage for timing results and outputs
    backend_times = {backend: [] for backend in backends}
    outputs = {}
    for cur_backend in backends:
        if run_refcheck:
            outputs[cur_backend] = run_backend(
                cur_backend, a_fp8, b_fp8, a_scale, b_scale, m_indptr
            ).detach()
        backend_times[cur_backend] = bench_gpu_time(
            fn=run_backend,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            sleep_after_run=True,  # GEMMs are very MMA-heavy, so prefer sleep to reduce throttling.
            enable_cupti=args.use_cupti,
            # DeepGEMM re-loads its cubin on every call (flashinfer/deep_gemm.py
            # load() looks up RUNTIME_CACHE without the ".cubin" suffix it
            # stores under), which CUDA-graph capture cannot record; time it
            # eagerly. CUPTI still reports kernel time only.
            use_cuda_graph=is_cuda_graph_compatible and cur_backend != "deepgemm",
            cold_l2_cache=True,
            input_args=(cur_backend, a_fp8, b_fp8, a_scale, b_scale, m_indptr),
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
                        f"[ERROR] Output tensor mismatch from backend {tested_backends[i]}"
                    )
                    if not args.allow_output_mismatch:
                        raise AssertionError(
                            f"[ERROR] Backend {tested_backends[i]} output mismatch with {num_different_elements} elements"
                        )

    for backend in backends:
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])
            problem_flops = 2 * m * n * k * group_size
            problem_bytes = (
                group_size * m * k + group_size * n * k
            ) * torch.float8_e4m3fn.itemsize + (group_size * m * n) * out_dtype.itemsize
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
            cur_res["n"] = n
            cur_res["k"] = k
            cur_res["group_size"] = group_size
            cur_res["tile_size"] = tile_size
            cur_res["scale_major_mode"] = scale_major_mode
            cur_res["out_dtype"] = out_dtype
            cur_res["mma_sm"] = mma_sm
            cur_res["backend"] = backend
            cur_res["case_tag"] = args.case_tag
            res.append(cur_res)
    return res


def testBmmFp8(args):
    """
    Test bmm_fp8 API.

    This test:
    1. Generates random input tensors
    2. Quantizes input tensors to FP8
    3. Runs bmm_fp8
    4. Runs reference check
    5. Measures performance metrics (TFLOPS, TB/sec)

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    warn_if_pdl_unsupported(args, args.routine)
    if args.verbose >= 1:
        print("[INFO] Running testBmmFp8")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    ## Parse input arguments
    backends = args.backends
    batch_size = args.batch_size
    m = args.m
    n = args.n
    k = args.k
    input_dtype = args.input_dtype
    mat2_dtype = args.mat2_dtype
    res_dtype = args.out_dtype
    backends = args.backends
    is_cuda_graph_compatible = not args.no_cuda_graph
    run_refcheck = args.refcheck
    autotune_supported_backends = [
        "cutlass",
        "cute-dsl",
        "cudnn",
        "cublas",
        "auto",
    ]
    res = []

    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)
    if input_dtype not in [torch.float8_e4m3fn, torch.float8_e5m2]:
        raise ValueError(
            f"Unsupported input dtype: {input_dtype}. Supported dtypes are fp8_e4m3 and fp8_e5m2."
        )

    mat2_dtype = dtype_str_to_torch_dtype(args.mat2_dtype)
    if mat2_dtype not in [torch.float8_e4m3fn, torch.float8_e5m2]:
        raise ValueError(
            f"Unsupported mat2 dtype: {mat2_dtype}. Supported dtypes are fp8_e4m3 and fp8_e5m2."
        )

    res_dtype = dtype_str_to_torch_dtype(args.out_dtype)
    if res_dtype not in [torch.bfloat16, torch.float16]:
        raise ValueError(
            f"Unsupported res dtype: {res_dtype}. Supported dtypes are bfloat16 and float16."
        )
    ## Done parsing input arguments

    ## Prepare input tensors
    input = torch.randn([batch_size, m, k], device=device, dtype=torch.bfloat16)
    input_fp8, input_inv_s = to_float8(input, dtype=input_dtype)

    mat2 = torch.randn(
        [batch_size, n, k], device=device, dtype=torch.bfloat16
    ).transpose(-2, -1)
    mat2_fp8, mat2_inv_s = to_float8(mat2, dtype=mat2_dtype)

    if args.verbose >= 2:
        print(f"[VVERBOSE] {input_fp8.shape = }")
        print(f"[VVERBOSE] {input_fp8.dtype = }")
        print(f"[VVERBOSE] {mat2_fp8.shape = }")
        print(f"[VVERBOSE] {mat2_fp8.dtype = }")
        print(f"[VVERBOSE] {input_inv_s = }")
        print(f"[VVERBOSE] {input_inv_s.dtype = }")
        print(f"[VVERBOSE] {mat2_inv_s = }")
        print(f"[VVERBOSE] {mat2_inv_s.dtype = }")

    # Filter backends by the cc whitelist first (mirrors bmm_fp8's per-backend
    # @supported_compute_capability), then rely on bmm_fp8's
    # @backend_requirement shape checks by probing each backend with a trial call.
    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    backends_to_remove = []
    for backend in backends:
        if (
            getattr(args, "autotune", False)
            and backend not in autotune_supported_backends
        ):
            print(f"[INFO] {backend} backend does not support autotune")
            backends_to_remove.append(backend)
            continue

        try:
            flashinfer.gemm.bmm_fp8(
                A=input_fp8,
                B=mat2_fp8,
                A_scale=input_inv_s,
                B_scale=mat2_inv_s,
                dtype=res_dtype,
                backend=backend,
            )
        except Exception as e:
            print(
                f"[INFO] {backend} backend does not support this configuration: {type(e).__name__}: {e}"
            )
            backends_to_remove.append(backend)

    for backend in backends_to_remove:
        backends.remove(backend)

    if len(backends) == 0:
        print("[ERROR] No backends passed validation. Exiting.")
        return res

    def run_backend(backend, input_fp8, mat2_fp8, input_inv_s, mat2_inv_s):
        return flashinfer.gemm.bmm_fp8(
            A=input_fp8,
            B=mat2_fp8,
            A_scale=input_inv_s,
            B_scale=mat2_inv_s,
            dtype=res_dtype,
            backend=backend,
        )

    has_reference_output = False
    if run_refcheck:
        reference_output = torch.bmm(input, mat2)
        has_reference_output = True

    cache_path = getattr(args, "autotune_cache", None)
    if getattr(args, "autotune", False):
        warmup_iters = (
            args.dry_run_iters if args.dry_run_iters and args.dry_run_iters > 0 else 10
        )
        for cur_backend in backends:
            if cur_backend in autotune_supported_backends:
                if args.verbose >= 1:
                    print(f"[INFO] Autotune warmup for bmm_fp8: {warmup_iters} iters")
                with autotune(True, cache=cache_path):
                    for _ in range(warmup_iters):
                        run_backend(
                            cur_backend, input_fp8, mat2_fp8, input_inv_s, mat2_inv_s
                        )
    elif cache_path:
        with autotune(False, cache=cache_path):
            pass

    # Storage for timing results and outputs
    backend_times = {backend: [] for backend in backends}
    outputs = {}
    for cur_backend in backends:
        if run_refcheck:
            outputs[cur_backend] = run_backend(
                cur_backend, input_fp8, mat2_fp8, input_inv_s, mat2_inv_s
            ).detach()
        backend_times[cur_backend] = bench_gpu_time(
            fn=run_backend,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            sleep_after_run=True,
            enable_cupti=args.use_cupti,
            use_cuda_graph=is_cuda_graph_compatible,
            cold_l2_cache=True,
            input_args=(cur_backend, input_fp8, mat2_fp8, input_inv_s, mat2_inv_s),
        )

    tested_backends = list(outputs.keys())
    tested_outputs = list(outputs.values())
    if len(tested_backends) > 0:
        if run_refcheck and has_reference_output:
            if reference_output.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                print(
                    "[INFO] Reference output is FP8. Converting to float32 for reference check."
                )
                reference_output = reference_output.to(torch.float32)
                tested_outputs = [output.to(torch.float32) for output in tested_outputs]
            for i in range(len(tested_backends)):
                # fp32: a float16 output against the bf16 reference otherwise
                # overflows the norms on large problems (cos_sim reads 0).
                cos_sim = F.cosine_similarity(
                    reference_output.float().reshape(-1),
                    tested_outputs[i].float().reshape(-1),
                    dim=0,
                )
                if cos_sim < 0.99:
                    print(
                        f"[ERROR] Output tensor mismatch between backends {tested_backends[0]} and {tested_backends[i]}"
                    )
                    if not args.allow_output_mismatch:
                        raise AssertionError(
                            f"[ERROR] Backend {tested_backends[i]} output mismatch with cos_sim={cos_sim}"
                        )

    for backend in backends:
        backend_name = backend + (
            "_autotune"
            if (
                getattr(args, "autotune", False)
                and backend in autotune_supported_backends
            )
            else ""
        )
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])
            problem_flops = 2 * m * n * k * batch_size
            problem_bytes = (
                m * k * input_dtype.itemsize
                + n * k * mat2_dtype.itemsize
                + m * n * res_dtype.itemsize
            )
            tflops = problem_flops / (10**9 * median_time)  # in TFLOPs/sec
            tb_per_sec = problem_bytes / (10**9 * median_time)  # in TB/sec
            print_perf_metrics(backend_name, median_time, std_time, tflops, tb_per_sec)

            if args.output_path is not None:
                cur_res = defaultdict(str)
                cur_res["batch_size"] = batch_size
                cur_res["routine"] = args.routine
                cur_res["median_time"] = median_time
                cur_res["std_time"] = std_time
                cur_res["tflops"] = tflops
                cur_res["tb_per_sec"] = tb_per_sec
                cur_res["m"] = m
                cur_res["n"] = n
                cur_res["k"] = k
                cur_res["input_dtype"] = input_dtype
                cur_res["mat2_dtype"] = mat2_dtype
                cur_res["out_dtype"] = res_dtype
                cur_res["backend"] = backend_name
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res


def testMmFp8(args):
    """
    Test mm_fp8 API.

    This test:
    1. Generates random input tensors and quantizes them to FP8
    2. Pre-shuffles the weight tensor into the low-latency block layout for
       trtllm_low_latency; every other backend takes the unprocessed (n, k) weight
    3. Runs mm_fp8
    4. Runs reference check against the unquantized bf16 GEMM
    5. Measures performance metrics (TFLOPS, TB/sec)

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    warn_if_pdl_unsupported(args, args.routine)
    if args.verbose >= 1:
        print("[INFO] Running testMmFp8")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    ## Parse input arguments
    backends = list(args.backends)
    m = args.m
    n = args.n
    k = args.k
    is_cuda_graph_compatible = not args.no_cuda_graph
    run_refcheck = args.refcheck
    autotune_supported_backends = [
        "trtllm_low_latency",
        "cublas",
        "cudnn",
        "cutlass",
        "prims-ts",
        "auto",
    ]
    res = []

    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)
    if input_dtype != torch.float8_e4m3fn:
        raise ValueError(
            f"Unsupported input dtype: {input_dtype}. mm_fp8 only supports fp8_e4m3."
        )

    mat2_dtype = dtype_str_to_torch_dtype(args.mat2_dtype)
    if mat2_dtype != torch.float8_e4m3fn:
        raise ValueError(
            f"Unsupported mat2 dtype: {mat2_dtype}. mm_fp8 only supports fp8_e4m3."
        )

    res_dtype = dtype_str_to_torch_dtype(args.out_dtype)
    if res_dtype != torch.bfloat16:
        raise ValueError(
            f"Unsupported res dtype: {res_dtype}. mm_fp8 only supports bfloat16."
        )

    if args.batch_size != 1:
        raise ValueError("mm_fp8 is a 2D GEMM; only --batch_size 1 is supported.")
    if k % 128 != 0:
        raise ValueError(f"mm_fp8 requires k % 128 == 0, got k = {k}")
    if n % 32 != 0:
        raise ValueError(f"mm_fp8 requires n % 32 == 0, got n = {n}")
    ## Done parsing input arguments

    ## Prepare input tensors
    input_tensor = torch.randn([m, k], device=device, dtype=torch.bfloat16)
    input_fp8, input_inv_s = to_float8(input_tensor, dtype=input_dtype)

    mat2 = torch.randn([n, k], device=device, dtype=torch.bfloat16)
    mat2_fp8, mat2_inv_s = to_float8(mat2, dtype=mat2_dtype)

    # Pre-shuffle the weights into the (k // 128, n, 128) low-latency block
    # layout. The dict memoizes the (expensive) permutation indices; a fresh one
    # is fine since weight prep runs once here, outside the timed region.
    mat2_prepared = flashinfer.prepare_low_latency_gemm_weights(mat2_fp8, {})
    alpha = (input_inv_s * mat2_inv_s).float()  # a_scale * b_scale

    if args.verbose >= 2:
        print(f"[VVERBOSE] {input_fp8.shape = }")
        print(f"[VVERBOSE] {input_fp8.dtype = }")
        print(f"[VVERBOSE] {mat2_prepared.shape = }")
        print(f"[VVERBOSE] {mat2_prepared.dtype = }")
        print(f"[VVERBOSE] {input_inv_s = }")
        print(f"[VVERBOSE] {mat2_inv_s = }")

    def weight_for(backend):
        # Only trtllm_low_latency reads the prepared layout; "auto" follows the
        # weight it is given, and gets the unprocessed one here.
        return mat2_prepared if backend == "trtllm_low_latency" else mat2_fp8

    # prims-ts is flashinfer.gemm.fp8_linear (PrimsTS dense GEMM): the same
    # product with per-token / per-channel fp32 scales, so the per-tensor
    # scales are broadcast once here, outside the timed region.
    prims_ts_scales = (
        (
            input_inv_s.float().reshape(1).expand(m).contiguous(),
            mat2_inv_s.float().reshape(1).expand(n).contiguous(),
        )
        if "prims-ts" in backends
        else None
    )

    def run_backend(backend, input_fp8, mat2, alpha):
        if backend == "prims-ts":
            return flashinfer.gemm.fp8_linear(
                input_fp8, mat2, *prims_ts_scales, out_dtype=res_dtype
            )
        return flashinfer.mm_fp8(
            input_fp8,
            mat2,
            alpha,
            out_dtype=res_dtype,
            backend=backend,
        )

    # Filter by the cc whitelist, then probe each backend with a trial call
    # (mm_fp8's @backend_requirement raises on unsupported layouts/shapes).
    # NOTE: unlike some other testers, backends without autotune support are
    # kept when --autotune is set (they just skip the warmup loop below and
    # are reported without the "_autotune" suffix).
    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    backends = _drop_failing_backends(
        backends, lambda bk: run_backend(bk, input_fp8, weight_for(bk), alpha)
    )
    if len(backends) == 0:
        print("[ERROR] No backends passed validation. Exiting.")
        return res

    has_reference_output = False
    if run_refcheck:
        reference_output = torch.mm(input_tensor, mat2.T)
        has_reference_output = True

    cache_path = getattr(args, "autotune_cache", None)
    if getattr(args, "autotune", False):
        warmup_iters = (
            args.dry_run_iters if args.dry_run_iters and args.dry_run_iters > 0 else 10
        )
        for cur_backend in backends:
            if cur_backend in autotune_supported_backends:
                if args.verbose >= 1:
                    print(f"[INFO] Autotune warmup for mm_fp8: {warmup_iters} iters")
                with autotune(True, cache=cache_path):
                    for _ in range(warmup_iters):
                        run_backend(
                            cur_backend, input_fp8, weight_for(cur_backend), alpha
                        )
    elif cache_path:
        with autotune(False, cache=cache_path):
            pass

    # Storage for timing results and outputs
    backend_times = {backend: [] for backend in backends}
    outputs = {}
    for cur_backend in backends:
        if run_refcheck:
            outputs[cur_backend] = run_backend(
                cur_backend, input_fp8, weight_for(cur_backend), alpha
            ).detach()
        backend_times[cur_backend] = bench_gpu_time(
            fn=run_backend,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            sleep_after_run=True,
            enable_cupti=args.use_cupti,
            use_cuda_graph=is_cuda_graph_compatible,
            cold_l2_cache=True,
            input_args=(cur_backend, input_fp8, weight_for(cur_backend), alpha),
        )

    tested_backends = list(outputs.keys())
    tested_outputs = list(outputs.values())
    if len(tested_backends) > 0:
        if run_refcheck and has_reference_output:
            for i in range(len(tested_backends)):
                cos_sim = F.cosine_similarity(
                    reference_output.reshape(-1),
                    tested_outputs[i].reshape(-1),
                    dim=0,
                )
                if torch.isnan(cos_sim) or cos_sim < 0.99:
                    print(
                        f"[ERROR] Output tensor mismatch between reference and backend {tested_backends[i]}"
                    )
                    if not args.allow_output_mismatch:
                        raise AssertionError(
                            f"[ERROR] Backend {tested_backends[i]} output mismatch with cos_sim={cos_sim}"
                        )

    for backend in backends:
        backend_name = backend + (
            "_autotune"
            if (
                getattr(args, "autotune", False)
                and backend in autotune_supported_backends
            )
            else ""
        )
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])
            problem_flops = 2 * m * n * k
            problem_bytes = (
                m * k * input_dtype.itemsize
                + n * k * mat2_dtype.itemsize
                + m * n * res_dtype.itemsize
            )
            tflops = problem_flops / (10**9 * median_time)  # in TFLOPs/sec
            tb_per_sec = problem_bytes / (10**9 * median_time)  # in TB/sec
            print_perf_metrics(backend_name, median_time, std_time, tflops, tb_per_sec)

            if args.output_path is not None:
                cur_res = defaultdict(str)
                cur_res["batch_size"] = 1
                cur_res["routine"] = args.routine
                cur_res["median_time"] = median_time
                cur_res["std_time"] = std_time
                cur_res["tflops"] = tflops
                cur_res["tb_per_sec"] = tb_per_sec
                cur_res["m"] = m
                cur_res["n"] = n
                cur_res["k"] = k
                cur_res["input_dtype"] = input_dtype
                cur_res["mat2_dtype"] = mat2_dtype
                cur_res["out_dtype"] = res_dtype
                cur_res["backend"] = backend_name
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res


def testBmmMxfp8(args):
    """
    Test bmm_mxfp8 API.

    This test:
    1. Generates random input tensors
    2. Quantizes input tensors to MXFP8
    3. Runs bmm_mxfp8
    4. Runs reference check
    5. Measures performance metrics (TFLOPS, TB/sec)

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    warn_if_pdl_unsupported(args, args.routine)
    if args.verbose >= 1:
        print("[INFO] Running testBmmMxfp8")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    ## Parse input arguments
    backends = args.backends
    batch_size = args.batch_size
    m = args.m
    n = args.n
    k = args.k
    res_dtype = args.out_dtype
    is_cuda_graph_compatible = not args.no_cuda_graph
    run_refcheck = args.refcheck
    autotune_supported_backends = [
        "cudnn",
    ]
    res = []

    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    res_dtype = dtype_str_to_torch_dtype(args.out_dtype)
    if res_dtype not in [torch.bfloat16, torch.float16]:
        raise ValueError(
            f"Unsupported res dtype: {res_dtype}. Supported dtypes are bfloat16 and float16."
        )
    ## Done parsing input arguments

    if getattr(args, "autotune", False):
        backends_to_remove = []
        for cur_backend in backends:
            if cur_backend not in autotune_supported_backends:
                print(f"[INFO] {cur_backend} backend does not support autotune")
                backends_to_remove.append(cur_backend)
        for cur_backend in backends_to_remove:
            backends.remove(cur_backend)

    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    ## Prepare input tensors
    input = torch.randn([batch_size, m, k], device=device, dtype=torch.bfloat16)
    input_mxfp8, input_scale = mxfp8_quantize(input, is_sf_swizzled_layout=True)

    # Quantize [b, n, k] weights, then pass their [b, k, n] transpose as B.
    # Keep the transpose as a view so the K dimension remains contiguous.
    mat2_weight = torch.randn([batch_size, n, k], device=device, dtype=torch.bfloat16)
    mat2_mxfp8, mat2_scale = mxfp8_quantize(mat2_weight, is_sf_swizzled_layout=True)
    # [b, k, n] views for the GEMM call and the reference.
    mat2_mxfp8 = mat2_mxfp8.transpose(-2, -1)
    mat2 = mat2_weight.transpose(-2, -1)

    if args.verbose >= 2:
        print(f"[VVERBOSE] {input_mxfp8.shape = }")
        print(f"[VVERBOSE] {input_mxfp8.dtype = }")
        print(f"[VVERBOSE] {mat2_mxfp8.shape = }")
        print(f"[VVERBOSE] {mat2_mxfp8.dtype = }")
        print(f"[VVERBOSE] {input_scale.shape = }")
        print(f"[VVERBOSE] {input_scale.dtype = }")
        print(f"[VVERBOSE] {mat2_scale.shape = }")
        print(f"[VVERBOSE] {mat2_scale.dtype = }")

    def run_backend(backend, input_mxfp8, mat2_mxfp8, input_scale, mat2_scale):
        # cutlass is the SM120/SM121 kernel (_cutlass_bmm_mxfp8_requirement).
        if backend in ["cudnn", "cutlass", "auto"]:
            return flashinfer.gemm.bmm_mxfp8(
                A=input_mxfp8,
                B=mat2_mxfp8,
                A_scale=input_scale,
                B_scale=mat2_scale,
                dtype=res_dtype,
                backend=backend,
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    backends = _drop_failing_backends(
        backends,
        lambda bk: run_backend(bk, input_mxfp8, mat2_mxfp8, input_scale, mat2_scale),
    )
    if len(backends) == 0:
        print("[ERROR] No backends passed validation. Exiting.")
        return res

    has_reference_output = False
    if run_refcheck:
        reference_output = torch.bmm(input, mat2)
        has_reference_output = True

    cache_path = getattr(args, "autotune_cache", None)
    if getattr(args, "autotune", False):
        warmup_iters = (
            args.dry_run_iters if args.dry_run_iters and args.dry_run_iters > 0 else 10
        )
        for cur_backend in backends:
            if cur_backend in autotune_supported_backends:
                if args.verbose >= 1:
                    print(f"[INFO] Autotune warmup for bmm_mxfp8: {warmup_iters} iters")
                with autotune(True, cache=cache_path):
                    for _ in range(warmup_iters):
                        run_backend(
                            cur_backend,
                            input_mxfp8,
                            mat2_mxfp8,
                            input_scale,
                            mat2_scale,
                        )
    elif cache_path:
        with autotune(False, cache=cache_path):
            pass

    # Storage for timing results and outputs
    backend_times = {backend: [] for backend in backends}
    outputs = {}
    for cur_backend in backends:
        if run_refcheck:
            outputs[cur_backend] = run_backend(
                cur_backend, input_mxfp8, mat2_mxfp8, input_scale, mat2_scale
            ).detach()
        backend_times[cur_backend] = bench_gpu_time(
            fn=run_backend,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            sleep_after_run=True,
            enable_cupti=args.use_cupti,
            use_cuda_graph=is_cuda_graph_compatible,
            cold_l2_cache=True,
            input_args=(cur_backend, input_mxfp8, mat2_mxfp8, input_scale, mat2_scale),
        )

    min_cos_sim = 0.9  # TODO: check if can be increased

    tested_backends = list(outputs.keys())
    tested_outputs = list(outputs.values())
    if len(tested_backends) > 0:
        if run_refcheck and has_reference_output:
            for i in range(len(tested_backends)):
                cos_sim = F.cosine_similarity(
                    reference_output.reshape(-1),
                    tested_outputs[i].reshape(-1),
                    dim=0,
                )
                if cos_sim < min_cos_sim:
                    print(
                        f"[ERROR] Output tensor mismatch between backends {tested_backends[0]} and {tested_backends[i]}"
                    )
                    if not args.allow_output_mismatch:
                        raise AssertionError(
                            f"[ERROR] Backend {tested_backends[i]} output mismatch with cos_sim={cos_sim}"
                        )

    for backend in backends:
        backend_name = backend + (
            "_autotune"
            if (
                getattr(args, "autotune", False)
                and backend in autotune_supported_backends
            )
            else ""
        )
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])
            problem_flops = 2 * m * n * k * batch_size
            # MXFP8 uses fp8_e4m3fn for data (1 byte) and uint8 for scales
            # Scale tensors are much smaller, so approximate as 1 byte per element for simplicity
            problem_bytes = (
                m * k * torch.float8_e4m3fn.itemsize
                + n * k * torch.float8_e4m3fn.itemsize
                + m * n * res_dtype.itemsize
            ) * batch_size
            tflops = problem_flops / (10**9 * median_time)  # in TFLOPs/sec
            tb_per_sec = problem_bytes / (10**9 * median_time)  # in TB/sec
            print_perf_metrics(backend_name, median_time, std_time, tflops, tb_per_sec)

            if args.output_path is not None:
                cur_res = defaultdict(str)
                cur_res["batch_size"] = batch_size
                cur_res["routine"] = args.routine
                cur_res["median_time"] = median_time
                cur_res["std_time"] = std_time
                cur_res["tflops"] = tflops
                cur_res["tb_per_sec"] = tb_per_sec
                cur_res["m"] = m
                cur_res["n"] = n
                cur_res["k"] = k
                cur_res["out_dtype"] = res_dtype
                cur_res["backend"] = backend_name
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res


def testMmFp4(args):
    """
    Test mm_fp4 API.

    This test:
    1. Generates random input tensors
    2. Quantizes input tensors to FP4
    3. Runs mm_fp4
    4. Runs reference check
    5. Measures performance metrics (TFLOPS, TB/sec)

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.verbose >= 1:
        print("[INFO] Running testMmFp4")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    ## Parse input arguments
    backends = args.backends
    m = args.m
    n = args.n
    k = args.k
    res_dtype = args.out_dtype
    backends = args.backends
    is_cuda_graph_compatible = not args.no_cuda_graph
    run_refcheck = args.refcheck
    use_128x4_sf_layout = args.use_128x4_sf_layout
    use_nvfp4 = args.use_nvfp4
    autotune_supported_backends = [
        "cudnn",
        "cutlass",
        "trtllm",
        "cute-dsl",
        "cutedsl_low_latency",
        "b12x",
        "prims-ts",
        "auto",
    ]
    res = []

    res_dtype = dtype_str_to_torch_dtype(args.out_dtype)
    if res_dtype not in [torch.bfloat16, torch.float16]:
        raise ValueError(
            f"Unsupported res dtype: {res_dtype}. Supported dtypes are bfloat16 and float16."
        )

    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    if "prims-ts" in backends and not (use_nvfp4 and use_128x4_sf_layout):
        # flashinfer.gemm.fp4_linear: NVFP4 with 128x4 block scales only.
        print(
            "[INFO] prims-ts only supports --use_nvfp4 with --use_128x4_sf_layout, removing prims-ts from backends"
        )
        backends.remove("prims-ts")
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    input = torch.randn([m, k], device=device, dtype=torch.bfloat16)
    mat2 = torch.randn([n, k], device=device, dtype=torch.bfloat16)
    a_sf_layout = (
        flashinfer.SfLayout.layout_128x4
        if use_128x4_sf_layout
        else flashinfer.SfLayout.layout_8x4
    )

    global_sf_input = (448 * 6) / input.float().abs().nan_to_num().max()
    global_sf_mat2 = (448 * 6) / mat2.float().abs().nan_to_num().max()

    if use_nvfp4:
        input_fp4, input_inv_s = flashinfer.nvfp4_quantize(
            input, global_sf_input, sfLayout=a_sf_layout, do_shuffle=False
        )
        mat2_fp4, mat2_inv_s = flashinfer.nvfp4_quantize(
            mat2,
            global_sf_mat2,
            sfLayout=flashinfer.SfLayout.layout_128x4,
            do_shuffle=False,
        )
    else:  # mxfp4
        input_fp4, input_inv_s = flashinfer.mxfp4_quantize(input)
        mat2_fp4, mat2_inv_s = flashinfer.mxfp4_quantize(mat2)

    # The trtllm backend consumes a shuffled weight/scale layout. Prepare it
    # only when that backend is requested: the shuffle requires n to be a
    # multiple of 128, and an unaligned n must not fail the other backends.
    mat2_fp4_trtllm, mat2_inv_s_trtllm = None, None
    if "trtllm" in backends:
        try:
            mat2_fp4_trtllm, mat2_inv_s_trtllm = flashinfer.nvfp4_quantize(
                mat2,
                global_sf_mat2,
                sfLayout=flashinfer.SfLayout.layout_128x4,
                do_shuffle=True,
            )
        except Exception as e:
            print(
                f"[INFO] trtllm backend does not support this configuration: {type(e).__name__}: {e}"
            )
            backends.remove("trtllm")

    if args.verbose >= 2:
        print(f"[VVERBOSE] {input_fp4.shape = }")
        print(f"[VVERBOSE] {input_fp4.dtype = }")
        print(f"[VVERBOSE] {mat2_fp4.shape = }")
        print(f"[VVERBOSE] {mat2_fp4.dtype = }")

    alpha = 1.0 / (global_sf_input * global_sf_mat2) if use_nvfp4 else None
    # TODO: for MXFP4, we don't need a global scale, we should change the compile interface to make
    # alpha optional.
    alpha_for_cute_dsl_mxfp4 = (
        torch.tensor([1.0], dtype=torch.float32, device=device)
        if not use_nvfp4
        else None
    )

    def alpha_for(backend):
        # NOTE: cute-dsl/b12x get no alpha for nvfp4 (pre-existing behaviour,
        # kept so existing rows stay comparable; the cos-sim refcheck is
        # scale-invariant).
        if backend in ("cute-dsl", "b12x"):
            return alpha_for_cute_dsl_mxfp4
        if backend == "cutedsl_low_latency":
            return alpha if use_nvfp4 else alpha_for_cute_dsl_mxfp4
        return alpha

    # prims-ts (flashinfer.gemm.fp4_linear) takes 1D 128x4 scale buffers and
    # host-float global dequant scales.
    prims_ts_globals = (
        (1.0 / float(global_sf_input), 1.0 / float(global_sf_mat2))
        if "prims-ts" in backends
        else None
    )
    # Completed preparing inputs. Now programmatically filter backends
    block_size = 16 if use_nvfp4 else 32
    backends_to_remove = []

    def run_backend(
        backend,
        input_fp4,
        mat2_fp4,
        mat2_fp4_trtllm,
        input_inv_s,
        mat2_inv_s,
        mat2_inv_s_trtllm,
    ):
        if backend == "prims-ts":
            return flashinfer.gemm.fp4_linear(
                input_fp4,
                input_inv_s.reshape(-1),
                prims_ts_globals[0],
                mat2_fp4,
                mat2_inv_s.reshape(-1),
                prims_ts_globals[1],
                out_dtype=res_dtype,
            )
        return flashinfer.gemm.mm_fp4(
            a=input_fp4,
            b=mat2_fp4.T if backend != "trtllm" else mat2_fp4_trtllm.T,
            a_descale=input_inv_s,
            b_descale=mat2_inv_s.T if backend != "trtllm" else mat2_inv_s_trtllm.T,
            alpha=alpha_for(backend),
            out_dtype=res_dtype,
            block_size=block_size,
            use_8x4_sf_layout=not use_128x4_sf_layout,
            backend=backend,
            use_nvfp4=use_nvfp4,
            enable_pdl=args.enable_pdl,
        )

    for backend in backends:
        # Skip autotune check for now (handled separately below)
        if (
            getattr(args, "autotune", False)
            and backend not in autotune_supported_backends
        ):
            print(f"[INFO] {backend} backend does not support autotune")
            backends_to_remove.append(backend)
            continue

        try:
            run_backend(
                backend,
                input_fp4,
                mat2_fp4,
                mat2_fp4_trtllm,
                input_inv_s,
                mat2_inv_s,
                mat2_inv_s_trtllm,
            )
        except Exception as e:
            print(
                f"[INFO] {backend} backend does not support this configuration: {type(e).__name__}: {e}"
            )
            backends_to_remove.append(backend)

    # Remove unsupported backends
    for backend in backends_to_remove:
        backends.remove(backend)

    if len(backends) == 0:
        print("[ERROR] No backends passed validation. Exiting.")
        return res

    has_reference_output = False
    if run_refcheck:
        reference_output = torch.mm(input, mat2.T)
        has_reference_output = True

    cache_path = getattr(args, "autotune_cache", None)
    if getattr(args, "autotune", False):
        warmup_iters = (
            args.dry_run_iters if args.dry_run_iters and args.dry_run_iters > 0 else 10
        )
        for cur_backend in backends:
            if args.verbose >= 1:
                print(f"[INFO] Autotune warmup for mm_fp4: {warmup_iters} iters")
            with autotune(True, cache=cache_path):
                for _ in range(warmup_iters):
                    run_backend(
                        cur_backend,
                        input_fp4,
                        mat2_fp4,
                        mat2_fp4_trtllm,
                        input_inv_s,
                        mat2_inv_s,
                        mat2_inv_s_trtllm,
                    )
    elif cache_path:
        with autotune(False, cache=cache_path):
            pass

    # Storage for timing results and outputs
    backend_times = {backend: [] for backend in backends}
    outputs = {}
    for cur_backend in backends:
        if run_refcheck:
            outputs[cur_backend] = run_backend(
                cur_backend,
                input_fp4,
                mat2_fp4,
                mat2_fp4_trtllm,
                input_inv_s,
                mat2_inv_s,
                mat2_inv_s_trtllm,
            ).detach()
        backend_times[cur_backend] = bench_gpu_time(
            fn=run_backend,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            sleep_after_run=True,
            enable_cupti=args.use_cupti,
            use_cuda_graph=is_cuda_graph_compatible,
            cold_l2_cache=True,
            input_args=(
                cur_backend,
                input_fp4,
                mat2_fp4,
                mat2_fp4_trtllm,
                input_inv_s,
                mat2_inv_s,
                mat2_inv_s_trtllm,
            ),
        )

    tested_backends = list(outputs.keys())
    tested_outputs = list(outputs.values())
    if len(tested_backends) > 0:
        if run_refcheck and has_reference_output:
            for i in range(len(tested_backends)):
                cos_sim = F.cosine_similarity(
                    reference_output.reshape(-1),
                    tested_outputs[i].reshape(-1),
                    dim=0,
                )
                if cos_sim < 0.97:
                    print(
                        f"[ERROR] Output tensor mismatch between backends {tested_backends[0]} and {tested_backends[i]}"
                    )
                    if not args.allow_output_mismatch:
                        raise AssertionError(
                            f"[ERROR] Backend {tested_backends[i]} output mismatch with cos_sim={cos_sim}"
                        )

    for backend in backends:
        backend_name = backend + (
            "_autotune"
            if (
                getattr(args, "autotune", False)
                and backend in autotune_supported_backends
            )
            else ""
        )
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])
            problem_flops = 2 * m * n * k
            problem_bytes = (
                m * k * 0.5 + n * k * 0.5 + m * n * res_dtype.itemsize
            )  # 0.5 for fp4
            tflops = problem_flops / (10**9 * median_time)  # in TFLOPs/sec
            tb_per_sec = problem_bytes / (10**9 * median_time)  # in TB/sec
            print_perf_metrics(backend_name, median_time, std_time, tflops, tb_per_sec)

            if args.output_path is not None:
                cur_res = defaultdict(str)
                cur_res["routine"] = args.routine
                cur_res["median_time"] = median_time
                cur_res["std_time"] = std_time
                cur_res["tflops"] = tflops
                cur_res["tb_per_sec"] = tb_per_sec
                cur_res["m"] = m
                cur_res["n"] = n
                cur_res["k"] = k
                cur_res["out_dtype"] = res_dtype
                cur_res["use_128x4_sf_layout"] = use_128x4_sf_layout
                cur_res["backend"] = backend_name
                cur_res["use_nvfp4"] = use_nvfp4
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res


# E2M1 (FP4) value table, signed (codes 0-7 positive, 8-15 negative), matching
# ``flashinfer.nvfp4_quantize``.
_E2M1_VALUES_FP32 = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)  # fmt: skip


def _dequantize_bf16_fp4_ref(b, b_descale, alpha, n, k, block_size):
    """PyTorch implementation of swizzled nvfp4 dequantization to fp32."""
    import torch
    from flashinfer.gemm.gemm_bf16_fp4 import _unswizzle_sf_128x4

    device = b.device
    k_sf = k // block_size
    lut = torch.tensor(_E2M1_VALUES_FP32, dtype=torch.float32, device=device)
    b_int = b.to(torch.int64)
    codes = torch.stack([b_int & 0xF, (b_int >> 4) & 0xF], dim=-1).reshape(n, k)
    values = lut[codes]
    sf = _unswizzle_sf_128x4(b_descale, n, k_sf).view(torch.float8_e4m3fn)
    sf_expanded = sf.to(torch.float32).repeat_interleave(block_size, dim=1)
    weight = values * sf_expanded
    if alpha is not None:
        weight = weight * alpha.to(torch.float32)
    return weight


def testMmBf16Fp4(args):
    """Benchmark mm_bf16_fp4 (bf16 activation x FP4 weight, bf16 x fp4).

    Weights are produced by ``flashinfer.nvfp4_quantize(sfLayout=layout_128x4)``
    -- the same format the new API expects.

    Constraints:
      * N must be divisible by 64 (the weight-prepack / kernel N tile).
        The 128x4 SF swizzle only pads N to 128; prepare_bf16_fp4_weights
        unswizzles the padded tail, so any N % 64 == 0 works.
      * K must be divisible by 16 (FP4 block size).
      * input_dtype must be bfloat16 (the only A dtype currently
        supported; fp16 deferred).

    Refcheck uses an fp32 dequant + matmul.
    """
    if args.verbose >= 1:
        print("[INFO] Running testMmBf16Fp4")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    m, n, k = args.m, args.n, args.k
    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)
    out_dtype = dtype_str_to_torch_dtype(args.out_dtype)
    backends = args.backends
    run_refcheck = args.refcheck
    is_cuda_graph_compatible = not args.no_cuda_graph

    if input_dtype != torch.bfloat16:
        raise ValueError(
            f"mm_bf16_fp4 benchmark requires input_dtype=bfloat16, got {args.input_dtype}"
        )
    if out_dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(
            f"mm_bf16_fp4 benchmark requires out_dtype in (bfloat16, float16), got {args.out_dtype}"
        )
    if n % 64 != 0:
        # N must be a multiple of the 64-wide N tile used by the weight
        # prepack and the cute-dsl kernel.  The 128x4 SF swizzle only *pads*
        # N to 128, and prepare_bf16_fp4_weights unswizzles the padded
        # tail, so any n % 64 == 0 works.
        raise ValueError("mm_bf16_fp4 benchmark requires n % 64 == 0")
    if k % 16 != 0:
        raise ValueError("mm_bf16_fp4 benchmark requires k % 16 == 0 (FP4 block size)")

    torch.manual_seed(args.random_seed)
    a = torch.randn((m, k), device=device, dtype=input_dtype) * 0.5
    w = torch.randn((n, k), device=device, dtype=input_dtype) * 0.1
    g_b = (448 * 6) / w.float().abs().nan_to_num().max()
    b_fp4, b_sf = flashinfer.nvfp4_quantize(
        w,
        g_b,
        sfLayout=flashinfer.SfLayout.layout_128x4,
        do_shuffle=False,
        backend="cute-dsl",
    )
    alpha = torch.tensor([1.0 / g_b.item()], device=device, dtype=torch.float32)

    if args.verbose >= 2:
        print(f"[VVERBOSE] {a.shape = } {a.dtype = }")
        print(f"[VVERBOSE] {b_fp4.shape = } {b_fp4.dtype = }")
        print(f"[VVERBOSE] {b_sf.shape = } {b_sf.dtype = }")

    # Per-backend prep + runner closures.  Prep is one-shot and not timed.
    backend_runners = {}

    def make_runner(b_p, sf_p, alpha_p, backend):
        def run(a):
            # mm_bf16_fp4: bf16 activation a against the prepared FP4
            # weight; b_p/sf_p are prepare_bf16_fp4_weights outputs.
            return flashinfer.mm_bf16_fp4(
                a,
                b_p,
                sf_p,
                alpha_p,
                backend=backend,
                out_dtype=out_dtype,
                block_size=16,
                enable_pdl=args.enable_pdl,
            )

        return run

    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    backends_to_remove = []
    for backend in backends:
        try:
            b_p, sf_p, alpha_p = flashinfer.prepare_bf16_fp4_weights(
                b_fp4, b_sf, alpha, backend=backend
            )
            runner = make_runner(b_p, sf_p, alpha_p, backend)
            runner(a)
            backend_runners[backend] = runner
        except Exception as e:
            print(
                f"[INFO] {backend} backend does not support this configuration: {type(e).__name__}: {e}"
            )
            backends_to_remove.append(backend)

    for backend in backends_to_remove:
        backends.remove(backend)

    if len(backends) == 0:
        print("[ERROR] No backends passed validation. Exiting.")
        return []

    autotune_supported_backends = ["cudnn", "cute-dsl", "cute-dsl-native"]
    cache_path = getattr(args, "autotune_cache", None)
    if getattr(args, "autotune", False):
        warmup_iters = (
            args.dry_run_iters if args.dry_run_iters and args.dry_run_iters > 0 else 10
        )
        for cur_backend in backends:
            if cur_backend in autotune_supported_backends:
                if args.verbose >= 1:
                    print(
                        f"[INFO] Autotune warmup for mm_bf16_fp4 {cur_backend}: "
                        f"{warmup_iters} iters"
                    )
                with autotune(True, cache=cache_path):
                    for _ in range(warmup_iters):
                        backend_runners[cur_backend](a)
    elif cache_path:
        with autotune(False, cache=cache_path):
            pass

    ref = None
    if run_refcheck:
        weight_fp32 = _dequantize_bf16_fp4_ref(b_fp4, b_sf, alpha, n, k, 16)
        ref = (a.float() @ weight_fp32.T).to(out_dtype)

    res = []
    flops = 2 * m * n * k
    bytes_accessed = (
        m * k * input_dtype.itemsize
        + (k // 2) * n  # FP4 weight (uint8, 2 codes / byte)
        + (k // 16) * n  # FP8-E4M3 per-block SF
        + m * n * out_dtype.itemsize
    )

    refcheck_tol = dict(rtol=1.5e-2, atol=1.5e-2)
    for backend in backends:
        runner = backend_runners[backend]
        if run_refcheck:
            out = runner(a)
            try:
                torch.testing.assert_close(out, ref, **refcheck_tol)
            except AssertionError as e:
                if args.allow_output_mismatch:
                    print(f"[WARNING] {backend} output mismatch vs fp32 ref: {e}")
                else:
                    raise

        timing = bench_gpu_time(
            fn=runner,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            sleep_after_run=True,  # GEMMs are very MMA-heavy, so prefer sleep to reduce throttling.
            enable_cupti=args.use_cupti,
            use_cuda_graph=is_cuda_graph_compatible,
            cold_l2_cache=True,
            input_args=(a,),
        )
        median_time = float(np.median(timing))
        std_time = float(np.std(timing))
        tflops = flops / median_time / 1e9
        tb_per_sec = bytes_accessed / median_time / 1e9
        backend_name = backend + (
            "_autotune"
            if (
                getattr(args, "autotune", False)
                and backend in autotune_supported_backends
            )
            else ""
        )
        print_perf_metrics(backend_name, median_time, std_time, tflops, tb_per_sec)
        res.append(
            {
                "routine": args.routine,
                "median_time": median_time,
                "std_time": std_time,
                "tflops": tflops,
                "tb_per_sec": tb_per_sec,
                "backend": backend_name,
                "resolved_backend": backend,
                "m": m,
                "n": n,
                "k": k,
                "input_dtype": str(input_dtype).split(".")[-1],
                "out_dtype": str(out_dtype).split(".")[-1],
                "case_tag": args.case_tag,
            }
        )
    return res


def testMmMxfp8(args):
    """
    Test mm_mxfp8 API.

    This test:
    1. Generates random input tensors
    2. Quantizes input tensors to MXFP8
    3. Runs mm_mxfp8
    4. Runs reference check
    5. Measures performance metrics (TFLOPS, TB/sec)

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    warn_if_pdl_unsupported(args, args.routine)
    if args.verbose >= 1:
        print("[INFO] Running testMmMxfp8")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    ## Parse input arguments
    backends = args.backends
    m = args.m
    n = args.n
    k = args.k
    res_dtype = args.out_dtype
    is_cuda_graph_compatible = not args.no_cuda_graph
    run_refcheck = args.refcheck
    autotune_supported_backends = [
        "cutlass",
        "cute-dsl",
        "cutedsl_low_latency",
        "trtllm",
        "cudnn",
        "b12x",
        "auto",
    ]
    res = []

    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    res_dtype = dtype_str_to_torch_dtype(args.out_dtype)
    if res_dtype not in [torch.bfloat16, torch.float16]:
        raise ValueError(
            f"Unsupported res dtype: {res_dtype}. Supported dtypes are bfloat16 and float16."
        )
    ## Done parsing input arguments

    if getattr(args, "autotune", False):
        backends_to_remove = []
        for cur_backend in backends:
            if cur_backend not in autotune_supported_backends:
                print(f"[INFO] {cur_backend} backend does not support autotune")
                backends_to_remove.append(cur_backend)
        for cur_backend in backends_to_remove:
            backends.remove(cur_backend)

    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    inputs = {}
    input = torch.randn([m, k], device=device, dtype=torch.bfloat16)
    mat2 = torch.randn([n, k], device=device, dtype=torch.bfloat16)
    for backend in backends:
        ## Prepare input tensors
        # Every backend consumes swizzled scales; trtllm optionally uses 8x4 for A.
        if backend == "trtllm" and not args.use_128x4_sf_layout:
            sf_layout_input = flashinfer.SfLayout.layout_8x4
        else:
            sf_layout_input = flashinfer.SfLayout.layout_128x4
        input_mxfp8, input_scale = mxfp8_quantize(
            input, sf_swizzle_layout=sf_layout_input
        )
        # when using trtllm, the shuffle_matrix_sf_a will swizzle the layout.
        mat2_mxfp8, mat2_scale = mxfp8_quantize(
            mat2,
            is_sf_swizzled_layout=backend != "trtllm",
        )

        if backend == "trtllm":
            from flashinfer import shuffle_matrix_a, shuffle_matrix_sf_a

            mat2_mxfp8 = shuffle_matrix_a(mat2_mxfp8, 128).reshape(n, k)
            mat2_scale = shuffle_matrix_sf_a(
                mat2_scale.reshape(n, k // 32),
                128,
                num_elts_per_sf=32,
            )
            mat2_scale = mat2_scale.t()

        if args.verbose >= 2:
            print(f"[VERBOSE] {backend}: {input_mxfp8.shape = }")
            print(f"[VERBOSE] {backend}: {input_mxfp8.dtype = }")
            print(f"[VERBOSE] {backend}: {mat2_mxfp8.shape = }")
            print(f"[VERBOSE] {backend}: {mat2_mxfp8.dtype = }")
            print(f"[VERBOSE] {backend}: {input_scale.shape = }")
            print(f"[VERBOSE] {backend}: {input_scale.dtype = }")
            print(f"[VERBOSE] {backend}: {mat2_scale.shape = }")
            print(f"[VERBOSE] {backend}: {mat2_scale.dtype = }")
        inputs[backend] = (input_mxfp8, mat2_mxfp8, input_scale, mat2_scale)

    def run_backend(
        backend: str,
        inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        assert backend in [
            "cutlass",
            "trtllm",
            "cute-dsl",
            "cutedsl_low_latency",
            "cudnn",
            "b12x",
            "auto",
        ], f"Unsupported backend: {backend}"
        input_mxfp8, mat2_mxfp8, input_scale, mat2_scale = inputs
        return flashinfer.gemm.mm_mxfp8(
            a=input_mxfp8,
            b=mat2_mxfp8.t(),  # mm_mxfp8 expects b.t()
            a_descale=input_scale,
            b_descale=mat2_scale,
            out_dtype=res_dtype,
            backend=backend,
            use_8x4_sf_layout=backend == "trtllm" and not args.use_128x4_sf_layout,
        )

    # The cc whitelist cannot express shape limits (cutedsl_low_latency:
    # m <= 8 and k % 128 == 0; b12x: k % 128 == 0); probe and drop.
    backends = _drop_failing_backends(backends, lambda bk: run_backend(bk, inputs[bk]))
    if len(backends) == 0:
        print("[ERROR] No backends passed validation. Exiting.")
        return res

    has_reference_output = False
    if run_refcheck:
        reference_output = torch.mm(input, mat2.t())
        has_reference_output = True

    cache_path = getattr(args, "autotune_cache", None)
    if getattr(args, "autotune", False):
        warmup_iters = (
            args.dry_run_iters if args.dry_run_iters and args.dry_run_iters > 0 else 10
        )
        for cur_backend in backends:
            if cur_backend in autotune_supported_backends:
                if args.verbose >= 1:
                    print(f"[INFO] Autotune warmup for mm_mxfp8: {warmup_iters} iters")
                with autotune(True, cache=cache_path):
                    for _ in range(warmup_iters):
                        run_backend(
                            cur_backend,
                            inputs[cur_backend],
                        )
    elif cache_path:
        with autotune(False, cache=cache_path):
            pass

    # Storage for timing results and outputs
    backend_times = {backend: [] for backend in backends}
    outputs = {}
    for cur_backend in backends:
        if run_refcheck:
            outputs[cur_backend] = run_backend(
                cur_backend, inputs[cur_backend]
            ).detach()
        backend_times[cur_backend] = bench_gpu_time(
            fn=run_backend,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            sleep_after_run=True,
            enable_cupti=args.use_cupti,
            use_cuda_graph=is_cuda_graph_compatible,
            cold_l2_cache=True,
            input_args=(cur_backend, inputs[cur_backend]),
        )

    # Minimum cosine similarity for swizzled layout
    min_cos_sim = 0.98

    tested_backends = list(outputs.keys())
    tested_outputs = list(outputs.values())
    if len(tested_backends) > 0:
        if run_refcheck and has_reference_output:
            for i in range(len(tested_backends)):
                cos_sim = F.cosine_similarity(
                    reference_output.reshape(-1),
                    tested_outputs[i].reshape(-1),
                    dim=0,
                )
                if cos_sim < min_cos_sim:
                    print(
                        "[ERROR] Output tensor mismatch between reference "
                        f"{tested_backends[0]} and backend {tested_backends[i]}"
                    )
                    if not args.allow_output_mismatch:
                        raise AssertionError(
                            "[ERROR] Output tensor mismatch between reference "
                            f"{tested_backends[0]} and backend {tested_backends[i]} "
                            f"with {cos_sim=} (expected >= {min_cos_sim})"
                        )

    for backend in backends:
        backend_name = backend + (
            "_autotune"
            if (
                getattr(args, "autotune", False)
                and backend in autotune_supported_backends
            )
            else ""
        )
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])
            problem_flops = 2 * m * n * k
            # MXFP8 uses fp8_e4m3fn for data (1 byte) and uint8 for scales
            # Scale tensors are much smaller, so approximate as 1 byte per element for simplicity
            problem_bytes = (
                m * k * torch.float8_e4m3fn.itemsize
                + n * k * torch.float8_e4m3fn.itemsize
                + m * n * res_dtype.itemsize
            )
            tflops = problem_flops / (10**9 * median_time)  # in TFLOPs/sec
            tb_per_sec = problem_bytes / (10**9 * median_time)  # in TB/sec
            print_perf_metrics(backend_name, median_time, std_time, tflops, tb_per_sec)

            if args.output_path is not None:
                cur_res = defaultdict(str)
                cur_res["routine"] = args.routine
                cur_res["median_time"] = median_time
                cur_res["std_time"] = std_time
                cur_res["tflops"] = tflops
                cur_res["tb_per_sec"] = tb_per_sec
                cur_res["m"] = m
                cur_res["n"] = n
                cur_res["k"] = k
                cur_res["out_dtype"] = res_dtype
                cur_res["backend"] = backend_name
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res


def testMmBf16(args):
    """
    Test mm_bf16 API.

    This test:
    1. Generates random BF16 input tensors
    2. Runs mm_bf16 with specified backend
    3. Runs reference check (torch.mm)
    4. Measures performance metrics (TFLOPS, TB/sec)

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.verbose >= 1:
        print("[INFO] Running testMmBf16")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    ## Parse input arguments
    backends = args.backends
    m = args.m
    n = args.n
    k = args.k
    use_bias = getattr(args, "bias", False)
    use_pdl = getattr(args, "enable_pdl", False)
    is_cuda_graph_compatible = not args.no_cuda_graph
    run_refcheck = args.refcheck
    autotune_supported_backends = [
        "cudnn",
        "cutlass",
        "tgv",
        "cublaslt",
        "tinygemm",
        "cute-dsl",
        "auto",
    ]
    res = []

    out_dtype = dtype_str_to_torch_dtype(args.out_dtype)
    if out_dtype not in [torch.bfloat16, torch.float16, torch.float32]:
        raise ValueError(
            f"Unsupported output dtype: {args.out_dtype}. Supported dtypes are bfloat16, float16, and float32."
        )

    ## Prepare input tensors
    # a: (m, k), row-major
    a = torch.randn([m, k], device=device, dtype=torch.bfloat16)
    # b: (n, k) then transpose to (k, n) for column-major layout
    b = torch.randn([n, k], device=device, dtype=torch.bfloat16).transpose(-2, -1)

    bias = None
    if use_bias:
        bias = torch.randn([n], device=device, dtype=torch.bfloat16)

    if args.verbose >= 2:
        print(f"[VVERBOSE] {a.shape = }")
        print(f"[VVERBOSE] {a.dtype = }")
        print(f"[VVERBOSE] {b.shape = }")
        print(f"[VVERBOSE] {b.dtype = }")
        if bias is not None:
            print(f"[VVERBOSE] {bias.shape = }")
            print(f"[VVERBOSE] {bias.dtype = }")
        print(f"[VVERBOSE] {use_pdl = }")

    # Programmatically filter backends: cc whitelist, then a trial call.
    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    backends_to_remove = []
    for backend in backends:
        # Skip autotune check for now (handled separately below)
        if (
            getattr(args, "autotune", False)
            and backend not in autotune_supported_backends
        ):
            print(f"[INFO] {backend} backend does not support autotune")
            backends_to_remove.append(backend)
            continue

        try:
            flashinfer.mm_bf16(
                a=a,
                b=b,
                bias=bias,
                pdl=use_pdl,
                out_dtype=out_dtype,
                backend=backend,
            )
        except Exception as e:
            print(
                f"[INFO] {backend} backend does not support this configuration: {type(e).__name__}: {e}"
            )
            backends_to_remove.append(backend)

    # Remove unsupported backends
    for backend in backends_to_remove:
        backends.remove(backend)

    if len(backends) == 0:
        print("[ERROR] No backends passed validation. Exiting.")
        return res

    def run_backend(backend, a, b, bias, use_pdl, out_dtype):
        if backend in [
            "cudnn",
            "cutlass",
            "tgv",
            "cublaslt",
            "tinygemm",
            "cutile",
            "cute-dsl",
            "auto",
        ]:
            return flashinfer.mm_bf16(
                a=a,
                b=b,
                bias=bias,
                pdl=use_pdl,
                out_dtype=out_dtype,
                backend=backend,
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    has_reference_output = False
    reference_output_base = None
    if run_refcheck:
        reference_output_base = torch.mm(a, b).to(out_dtype)
        has_reference_output = True

    cache_path = getattr(args, "autotune_cache", None)
    if getattr(args, "autotune", False):
        warmup_iters = (
            args.dry_run_iters if args.dry_run_iters and args.dry_run_iters > 0 else 10
        )
        for cur_backend in backends:
            if cur_backend in autotune_supported_backends:
                if args.verbose >= 1:
                    print(f"[INFO] Autotune warmup for mm_bf16: {warmup_iters} iters")
                with autotune(True, cache=cache_path):
                    for _ in range(warmup_iters):
                        run_backend(cur_backend, a, b, bias, use_pdl, out_dtype)
    elif cache_path:
        with autotune(False, cache=cache_path):
            pass

    # Storage for timing results and outputs
    backend_times = {backend: [] for backend in backends}
    outputs = {}
    for cur_backend in backends:
        if run_refcheck:
            outputs[cur_backend] = run_backend(
                cur_backend, a, b, bias, use_pdl, out_dtype
            ).detach()
        backend_times[cur_backend] = bench_gpu_time(
            fn=run_backend,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            sleep_after_run=True,
            enable_cupti=args.use_cupti,
            use_cuda_graph=is_cuda_graph_compatible,
            cold_l2_cache=True,
            input_args=(cur_backend, a, b, bias, use_pdl, out_dtype),
        )

    tested_backends = list(outputs.keys())
    tested_outputs = list(outputs.values())
    if len(tested_backends) > 0:
        if run_refcheck and has_reference_output:
            for i in range(len(tested_backends)):
                if (
                    tested_backends[i]
                    in (
                        "cudnn",
                        "tgv",
                        "tinygemm",
                        "cute-dsl",
                    )
                    and bias is not None
                ):
                    reference_output = reference_output_base + bias.unsqueeze(0).to(
                        out_dtype
                    )
                else:
                    reference_output = reference_output_base

                cos_sim = F.cosine_similarity(
                    reference_output.reshape(-1),
                    tested_outputs[i].reshape(-1),
                    dim=0,
                )
                if cos_sim < 0.99:
                    print(
                        f"[ERROR] Output tensor mismatch from backend {tested_backends[i]} with cos_sim={cos_sim}"
                    )
                    if not args.allow_output_mismatch:
                        raise AssertionError(
                            f"[ERROR] Backend {tested_backends[i]} output mismatch with cos_sim={cos_sim}"
                        )

    for backend in backends:
        backend_name = backend + (
            "_autotune"
            if (
                getattr(args, "autotune", False)
                and backend in autotune_supported_backends
            )
            else ""
        )
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])
            problem_flops = 2 * m * n * k
            if use_bias:
                problem_flops += m * n  # bias addition
            problem_bytes = (
                m * k * torch.bfloat16.itemsize
                + k * n * torch.bfloat16.itemsize
                + m * n * out_dtype.itemsize
            )
            if use_bias:
                problem_bytes += n * torch.bfloat16.itemsize
            tflops = problem_flops / (10**9 * median_time)  # in TFLOPs/sec
            tb_per_sec = problem_bytes / (10**9 * median_time)  # in TB/sec
            print_perf_metrics(backend_name, median_time, std_time, tflops, tb_per_sec)

            if args.output_path is not None:
                cur_res = defaultdict(str)
                cur_res["routine"] = args.routine
                cur_res["median_time"] = median_time
                cur_res["std_time"] = std_time
                cur_res["tflops"] = tflops
                cur_res["tb_per_sec"] = tb_per_sec
                cur_res["m"] = m
                cur_res["n"] = n
                cur_res["k"] = k
                cur_res["out_dtype"] = str(out_dtype)
                cur_res["backend"] = backend_name
                cur_res["bias"] = use_bias
                cur_res["enable_pdl"] = use_pdl
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res


def testTinygemmBf16(args):
    """
    Test tinygemm_bf16 API.

    This test:
    1. Generates random BF16 input tensors
    2. Runs tinygemm_bf16
    3. Runs reference check (F.linear)
    4. Measures performance metrics (TFLOPS, TB/sec)

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.verbose >= 1:
        print("[INFO] Running testTinygemmBf16")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    backends = list(args.backends)
    m = args.m
    n = args.n
    k = args.k
    use_bias = getattr(args, "bias", False)
    use_pdl = getattr(args, "enable_pdl", False)
    is_cuda_graph_compatible = not args.no_cuda_graph
    run_refcheck = args.refcheck
    res = []

    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)
    mat2_dtype = dtype_str_to_torch_dtype(args.mat2_dtype)
    out_dtype = dtype_str_to_torch_dtype(args.out_dtype)
    if input_dtype != torch.bfloat16 or mat2_dtype != torch.bfloat16:
        raise ValueError(
            "tinygemm_bf16 only supports bfloat16 input and weight tensors."
        )
    if out_dtype != torch.bfloat16:
        raise ValueError("tinygemm_bf16 only supports bfloat16 outputs.")

    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    from flashinfer.gemm import tinygemm_bf16

    input = torch.randn([m, k], device=device, dtype=torch.bfloat16)
    weight = torch.randn([n, k], device=device, dtype=torch.bfloat16)
    bias = torch.randn([n], device=device, dtype=torch.bfloat16) if use_bias else None
    outs = {
        backend: torch.empty([m, n], device=device, dtype=torch.bfloat16)
        for backend in backends
    }

    if args.verbose >= 2:
        print(f"[VVERBOSE] {input.shape = }")
        print(f"[VVERBOSE] {input.dtype = }")
        print(f"[VVERBOSE] {weight.shape = }")
        print(f"[VVERBOSE] {weight.dtype = }")
        if bias is not None:
            print(f"[VVERBOSE] {bias.shape = }")
            print(f"[VVERBOSE] {bias.dtype = }")
        print(f"[VVERBOSE] {use_pdl = }")

    backends_to_remove = []
    for backend in backends:
        if backend != "tinygemm":
            backends_to_remove.append(backend)
            continue
        try:
            tinygemm_bf16(input, weight, outs[backend], bias=bias, use_pdl=use_pdl)
        except Exception as e:
            print(
                f"[INFO] {backend} backend does not support this configuration: {type(e).__name__}: {e}"
            )
            backends_to_remove.append(backend)

    for backend in backends_to_remove:
        backends.remove(backend)
        outs.pop(backend, None)

    if len(backends) == 0:
        print("[ERROR] No backends passed validation. Exiting.")
        return res

    def run_backend(backend, input, weight, bias, out, use_pdl):
        if backend != "tinygemm":
            raise ValueError(f"Unsupported backend: {backend}")
        tinygemm_bf16(input, weight, out, bias=bias, use_pdl=use_pdl)
        return out

    has_reference_output = False
    if run_refcheck:
        reference_output = F.linear(
            input.float(),
            weight.float(),
            bias.float() if bias is not None else None,
        ).bfloat16()
        has_reference_output = True

    backend_times = {backend: [] for backend in backends}
    outputs = {}
    for cur_backend in backends:
        if run_refcheck:
            outputs[cur_backend] = (
                run_backend(
                    cur_backend, input, weight, bias, outs[cur_backend], use_pdl
                )
                .detach()
                .clone()
            )
        backend_times[cur_backend] = bench_gpu_time(
            fn=run_backend,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            sleep_after_run=True,
            enable_cupti=args.use_cupti,
            use_cuda_graph=is_cuda_graph_compatible,
            cold_l2_cache=True,
            input_args=(cur_backend, input, weight, bias, outs[cur_backend], use_pdl),
        )

    tested_backends = list(outputs.keys())
    tested_outputs = list(outputs.values())
    if len(tested_backends) > 0 and run_refcheck and has_reference_output:
        for i in range(len(tested_backends)):
            cos_sim = F.cosine_similarity(
                reference_output.reshape(-1).float(),
                tested_outputs[i].reshape(-1).float(),
                dim=0,
            )
            if cos_sim < 0.99:
                print(
                    f"[ERROR] Output tensor mismatch from backend {tested_backends[i]} with cos_sim={cos_sim}"
                )
                if not args.allow_output_mismatch:
                    raise AssertionError(
                        f"[ERROR] Backend {tested_backends[i]} output mismatch with cos_sim={cos_sim}"
                    )

    for backend in backends:
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])
            problem_flops = 2 * m * n * k
            if use_bias:
                problem_flops += m * n
            problem_bytes = (
                m * k * torch.bfloat16.itemsize
                + n * k * torch.bfloat16.itemsize
                + m * n * torch.bfloat16.itemsize
            )
            if use_bias:
                problem_bytes += n * torch.bfloat16.itemsize
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
                cur_res["n"] = n
                cur_res["k"] = k
                cur_res["input_dtype"] = input_dtype
                cur_res["mat2_dtype"] = mat2_dtype
                cur_res["out_dtype"] = out_dtype
                cur_res["backend"] = backend
                cur_res["bias"] = use_bias
                cur_res["enable_pdl"] = use_pdl
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res


def testBmmBf16(args):
    """
    Test bmm_bf16 API.

    This test:
    1. Generates random BF16 batched input tensors
    2. Runs bmm_bf16 with specified backend
    3. Runs reference check (torch.bmm)
    4. Measures performance metrics (TFLOPS, TB/sec)

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    warn_if_pdl_unsupported(args, args.routine)
    if args.verbose >= 1:
        print("[INFO] Running testBmmBf16")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    ## Parse input arguments
    backends = args.backends
    batch_size = args.batch_size
    m = args.m
    n = args.n
    k = args.k
    is_cuda_graph_compatible = not args.no_cuda_graph
    run_refcheck = args.refcheck
    autotune_supported_backends = ["cudnn", "cutlass", "auto"]
    res = []

    out_dtype = dtype_str_to_torch_dtype(args.out_dtype)
    if out_dtype not in [torch.bfloat16, torch.float16, torch.float32]:
        raise ValueError(
            f"Unsupported output dtype: {args.out_dtype}. Supported dtypes are bfloat16, float16, and float32."
        )

    ## Prepare input tensors
    # A: (batch_size, m, k), row-major
    A = torch.randn([batch_size, m, k], device=device, dtype=torch.bfloat16)
    # B: (batch_size, n, k) then transpose to (batch_size, k, n) for column-major layout
    B = torch.randn([batch_size, n, k], device=device, dtype=torch.bfloat16).transpose(
        -2, -1
    )

    if args.verbose >= 2:
        print(f"[VVERBOSE] {A.shape = }")
        print(f"[VVERBOSE] {A.dtype = }")
        print(f"[VVERBOSE] {B.shape = }")
        print(f"[VVERBOSE] {B.dtype = }")

    # Programmatically filter backends: cc whitelist, then a trial call.
    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    backends_to_remove = []
    for backend in backends:
        # Skip autotune check for now (handled separately below)
        if (
            getattr(args, "autotune", False)
            and backend not in autotune_supported_backends
        ):
            print(f"[INFO] {backend} backend does not support autotune")
            backends_to_remove.append(backend)
            continue

        try:
            flashinfer.bmm_bf16(
                A=A,
                B=B,
                out_dtype=out_dtype,
                backend=backend,
            )
        except Exception as e:
            print(
                f"[INFO] {backend} backend does not support this configuration: {type(e).__name__}: {e}"
            )
            backends_to_remove.append(backend)

    # Remove unsupported backends
    for backend in backends_to_remove:
        backends.remove(backend)

    if len(backends) == 0:
        print("[ERROR] No backends passed validation. Exiting.")
        return res

    def run_backend(backend, A, B, out_dtype):
        # Before cutile/tgv were accepted here they passed the trial call above
        # and then raised in the timed loop, killing the whole case.
        if backend in ["cudnn", "cutlass", "cutile", "tgv", "auto"]:
            return flashinfer.bmm_bf16(
                A=A,
                B=B,
                out_dtype=out_dtype,
                backend=backend,
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    has_reference_output = False
    if run_refcheck:
        reference_output = torch.bmm(A, B).to(out_dtype)
        has_reference_output = True

    if getattr(args, "autotune", False):
        warmup_iters = (
            args.dry_run_iters if args.dry_run_iters and args.dry_run_iters > 0 else 10
        )
        for cur_backend in backends:
            if cur_backend in autotune_supported_backends:
                if args.verbose >= 1:
                    print(f"[INFO] Autotune warmup for bmm_bf16: {warmup_iters} iters")
                with autotune(True):
                    for _ in range(warmup_iters):
                        run_backend(cur_backend, A, B, out_dtype)

    # Storage for timing results and outputs
    backend_times = {backend: [] for backend in backends}
    outputs = {}
    for cur_backend in backends:
        if run_refcheck:
            outputs[cur_backend] = run_backend(cur_backend, A, B, out_dtype).detach()
        backend_times[cur_backend] = bench_gpu_time(
            fn=run_backend,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            sleep_after_run=True,
            enable_cupti=args.use_cupti,
            use_cuda_graph=is_cuda_graph_compatible,
            cold_l2_cache=True,
            input_args=(cur_backend, A, B, out_dtype),
        )

    tested_backends = list(outputs.keys())
    tested_outputs = list(outputs.values())
    if len(tested_backends) > 0:
        if run_refcheck and has_reference_output:
            for i in range(len(tested_backends)):
                cos_sim = F.cosine_similarity(
                    reference_output.reshape(-1),
                    tested_outputs[i].reshape(-1),
                    dim=0,
                )
                if cos_sim < 0.99:
                    print(
                        f"[ERROR] Output tensor mismatch from backend {tested_backends[i]} with cos_sim={cos_sim}"
                    )
                    if not args.allow_output_mismatch:
                        raise AssertionError(
                            f"[ERROR] Backend {tested_backends[i]} output mismatch with cos_sim={cos_sim}"
                        )

    for backend in backends:
        backend_name = backend + (
            "_autotune"
            if (
                getattr(args, "autotune", False)
                and backend in autotune_supported_backends
            )
            else ""
        )
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])
            problem_flops = 2 * batch_size * m * n * k
            problem_bytes = batch_size * (
                m * k * torch.bfloat16.itemsize
                + k * n * torch.bfloat16.itemsize
                + m * n * out_dtype.itemsize
            )
            tflops = problem_flops / (10**9 * median_time)  # in TFLOPs/sec
            tb_per_sec = problem_bytes / (10**9 * median_time)  # in TB/sec
            print_perf_metrics(backend_name, median_time, std_time, tflops, tb_per_sec)

            if args.output_path is not None:
                cur_res = defaultdict(str)
                cur_res["batch_size"] = batch_size
                cur_res["routine"] = args.routine
                cur_res["median_time"] = median_time
                cur_res["std_time"] = std_time
                cur_res["tflops"] = tflops
                cur_res["tb_per_sec"] = tb_per_sec
                cur_res["m"] = m
                cur_res["n"] = n
                cur_res["k"] = k
                cur_res["out_dtype"] = str(out_dtype)
                cur_res["backend"] = backend_name
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res


# ---------------------------------------------------------------------------
# Grouped / specialized GEMM routines. They share one timing / refcheck / CSV
# tail (_bench_and_report) because their operand construction is per backend.
# ---------------------------------------------------------------------------


def _cos_sim(reference, output):
    return F.cosine_similarity(
        reference.float().reshape(-1), output.float().reshape(-1), dim=0
    )


def _bench_and_report(
    args,
    backends,
    run_fns,
    refchecks,
    problem_flops,
    problem_bytes,
    extra_fields,
    cos_sim_min=0.99,
    autotune_backends=(),
    no_graph_backends=(),
):
    """Shared timing / refcheck / CSV tail of the routines below.

    run_fns:   backend -> zero-argument callable launching the op.
    refchecks: backend -> callable(output) returning the cosine similarity of
               the backend's valid output against its reference, or None.
    no_graph_backends: backends timed without CUDA graphs (see callers).
    """
    res = []
    cache_path = getattr(args, "autotune_cache", None)
    if getattr(args, "autotune", False):
        warmup_iters = (
            args.dry_run_iters if args.dry_run_iters and args.dry_run_iters > 0 else 10
        )
        for backend in backends:
            if backend in autotune_backends:
                with autotune(True, cache=cache_path):
                    for _ in range(warmup_iters):
                        run_fns[backend]()
    elif cache_path:
        with autotune(False, cache=cache_path):
            pass

    backend_times = {}
    for backend in backends:
        if args.refcheck and refchecks.get(backend) is not None:
            cos_sim = refchecks[backend](run_fns[backend]())
            if torch.isnan(cos_sim) or cos_sim < cos_sim_min:
                print(
                    f"[ERROR] Output tensor mismatch from backend {backend} with cos_sim={cos_sim}"
                )
                if not args.allow_output_mismatch:
                    raise AssertionError(
                        f"[ERROR] Backend {backend} output mismatch with cos_sim={cos_sim}"
                    )
        backend_times[backend] = bench_gpu_time(
            fn=run_fns[backend],
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            sleep_after_run=True,  # GEMMs are very MMA-heavy, so prefer sleep to reduce throttling.
            enable_cupti=args.use_cupti,
            use_cuda_graph=not args.no_cuda_graph and backend not in no_graph_backends,
            cold_l2_cache=True,
        )

    for backend in backends:
        backend_name = backend + (
            "_autotune"
            if getattr(args, "autotune", False) and backend in autotune_backends
            else ""
        )
        if len(backend_times[backend]) == 0:
            continue
        median_time = np.median(backend_times[backend])
        std_time = np.std(backend_times[backend])
        tflops = problem_flops / (10**9 * median_time)  # in TFLOPs/sec
        tb_per_sec = problem_bytes / (10**9 * median_time)  # in TB/sec
        print_perf_metrics(backend_name, median_time, std_time, tflops, tb_per_sec)
        if args.output_path is not None:
            cur_res = defaultdict(str)
            cur_res["routine"] = args.routine
            cur_res["median_time"] = median_time
            cur_res["std_time"] = std_time
            cur_res["tflops"] = tflops
            cur_res["tb_per_sec"] = tb_per_sec
            cur_res["m"] = args.m
            cur_res["n"] = args.n
            cur_res["k"] = args.k
            cur_res["backend"] = backend_name
            cur_res["case_tag"] = args.case_tag
            cur_res.update(extra_fields)
            res.append(cur_res)
    return res


def _filter_and_probe(args, device, backends, run_fns):
    """cc whitelist, then a trial launch per backend (shape/dtype limits)."""
    backends = filter_backends_by_compute_capability(
        [b for b in backends if b in run_fns], args.routine, device
    )
    return _drop_failing_backends(backends, lambda bk: run_fns[bk]())


def _deepgemm_packed_scales(a_scale, b_scale, mn_a, n, k, num_groups, sfa_groups=None):
    """Pre-pack fp32 (1x128 / 128x128) scales into DeepGEMM's int UE8M0 layout
    once, outside the timed region. Given fp32 scales, the DeepGEMM entry points
    repack them on every call (several small kernels plus host work); packed
    int scales skip that transform, which is how serving stacks call them."""
    from flashinfer.deep_gemm import transform_sf_into_required_layout

    recipe = (1, 128, 128)
    sfa = transform_sf_into_required_layout(
        a_scale, mn=mn_a, k=k, recipe=recipe, num_groups=sfa_groups, is_sfa=True
    )
    sfb = transform_sf_into_required_layout(
        b_scale, mn=n, k=k, recipe=recipe, num_groups=num_groups, is_sfa=False
    )
    return sfa, sfb


# (k, n, out_dtype) -> the shape-specialized router GEMM (routergemm.py).
_ROUTER_GEMM_FNS = {
    (7168, 128, torch.bfloat16): "mm_M1_16_K7168_N128",
    (7168, 256, torch.float32): "mm_M1_16_K7168_N256",
    (6144, 256, torch.float32): "mm_M1_16_K6144_N256",
    (7168, 256, torch.bfloat16): "mm_M1_16_K7168_N256_bf16",
    (7168, 384, torch.float32): "mm_M1_16_K7168_N384",
    (7168, 384, torch.bfloat16): "mm_M1_16_K7168_N384_bf16",
    (7168, 896, torch.float32): "mm_M1_16_K7168_N896",
    (7168, 896, torch.bfloat16): "mm_M1_16_K7168_N896_bf16",
}


def testRouterGemm(args):
    """
    Test the shape-specialized MoE router GEMMs ``mm_M1_16_K{k}_N{n}[_bf16]``
    (flashinfer.gemm / flashinfer.dsv3_ops).

    bf16 activations (m in [1, 16]) times a column-major bf16 router weight.
    The function is picked by (k, n, out_dtype): float32 output selects the
    plain variant, bfloat16 the ``_bf16`` one (N128 is bf16-only). The API
    has no backend argument; the CSV label ``dsv3`` names the family. On
    SM100/SM103 the N128, N256 and K6144_N256 functions internally run a
    generated program instead of the DSv3 kernel (routergemm.py
    _router_gemm_module_for_device); that is library dispatch, not a
    selectable backend.
    """
    if args.verbose >= 1:
        print("[INFO] Running testRouterGemm")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")
    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    m, n, k = args.m, args.n, args.k
    out_dtype = dtype_str_to_torch_dtype(args.out_dtype)
    fn_name = _ROUTER_GEMM_FNS.get((k, n, out_dtype))
    if fn_name is None:
        print(
            f"[ERROR] No specialized router GEMM for k={k}, n={n}, out_dtype={args.out_dtype}. "
            f"Available (k, n, out_dtype): {sorted((kk, nn, str(dd)) for kk, nn, dd in _ROUTER_GEMM_FNS)}"
        )
        return []
    router_gemm = getattr(flashinfer.gemm, fn_name)

    a = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    b = torch.randn(n, k, device=device, dtype=torch.bfloat16).t()  # column-major
    out = torch.empty(m, n, device=device, dtype=out_dtype)

    def run():
        router_gemm(a, b, out, launch_with_pdl=args.enable_pdl)
        return out

    run_fns = {"dsv3": run}
    backends = _filter_and_probe(args, device, list(args.backends), run_fns)
    if len(backends) == 0:
        print("[ERROR] No backends passed validation. Exiting.")
        return []
    reference = a.float() @ b.float()
    refchecks = {backend: (lambda o: _cos_sim(reference, o)) for backend in backends}
    problem_flops = 2 * m * n * k
    problem_bytes = (m * k + n * k) * torch.bfloat16.itemsize + m * n * out.itemsize
    return _bench_and_report(
        args,
        backends,
        run_fns,
        refchecks,
        problem_flops,
        problem_bytes,
        {
            "input_dtype": str(torch.bfloat16),
            "out_dtype": str(out_dtype),
            "enable_pdl": args.enable_pdl,
        },
    )


def testGroupGemmMxfp4NtGroupwise(args):
    """
    Test group_gemm_mxfp4_nt_groupwise (= group_gemm_mxfp8_mxfp4_nt_groupwise).

    MXFP8 activations (e4m3 + per-32 UE8M0 scales) times MXFP4 expert weights,
    with group_size uniform segments of m rows (m_indptr). Scales come from
    the public mxfp8_quantize / mxfp4_quantize in the 128x4 swizzled layout,
    one quantization per group so each group's scale rows are padded to 128.
    Refcheck: cosine similarity against the unquantized bf16 product.
    """
    if args.verbose >= 1:
        print("[INFO] Running testGroupGemmMxfp4NtGroupwise")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")
    warn_if_pdl_unsupported(args, args.routine)
    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    m, n, k, group_size = args.m, args.n, args.k, args.group_size
    out_dtype = dtype_str_to_torch_dtype(args.out_dtype)
    if out_dtype not in [torch.bfloat16, torch.float16]:
        raise ValueError(f"Unsupported output dtype: {args.out_dtype}")
    if dtype_str_to_torch_dtype(args.input_dtype) != torch.float8_e4m3fn:
        raise ValueError(
            "group_gemm_mxfp4_nt_groupwise benchmarks MXFP8(e4m3) x MXFP4; use --input_dtype fp8_e4m3"
        )

    a_val = torch.randn(group_size * m, k, device=device, dtype=torch.bfloat16)
    b_val = (torch.randn(group_size, n, k, device=device) / np.sqrt(k)).to(
        torch.bfloat16
    )
    a_parts, a_scale_parts, b_parts, b_scale_parts = [], [], [], []
    for g in range(group_size):
        a_q, a_sf = mxfp8_quantize(
            a_val[g * m : (g + 1) * m].contiguous(), is_sf_swizzled_layout=True
        )
        a_parts.append(a_q)
        a_scale_parts.append(a_sf.reshape(-1, k // 32))
        b_q, b_sf = flashinfer.mxfp4_quantize(b_val[g].contiguous())
        b_parts.append(b_q.view(torch.uint8))
        b_scale_parts.append(b_sf.view(torch.uint8).reshape(-1, k // 32))
    a = torch.cat(a_parts)
    b = torch.stack(b_parts)
    b_scale = torch.stack(b_scale_parts)
    m_indptr = torch.arange(0, group_size + 1, dtype=torch.int32, device=device) * m
    # The kernel reads group g's A scales at row (m_indptr[g] + g * 127) // 128 * 128
    # (group_gemm_mxfp4_groupwise_sm100.cuh), not back to back; place each
    # group's 128-row-padded swizzled block there, zero-filled in between (as
    # tests/gemm/test_groupwise_scaled_gemm_mxfp4.py does).
    sf_offsets = [(g * m + g * 127) // 128 * 128 for g in range(group_size + 1)]
    a_scale = torch.zeros(sf_offsets[-1], k // 32, dtype=torch.uint8, device=device)
    for g, part in enumerate(a_scale_parts):
        a_scale[sf_offsets[g] : sf_offsets[g] + part.shape[0]] = part.view(torch.uint8)

    def run():
        return flashinfer.gemm.group_gemm_mxfp8_mxfp4_nt_groupwise(
            a,
            b,
            a_scale,
            b_scale,
            m_indptr,
            mma_sm=args.mma_sm,
            out_dtype=out_dtype,
        )

    run_fns = {"cutlass": run}
    backends = _filter_and_probe(args, device, list(args.backends), run_fns)
    if len(backends) == 0:
        print("[ERROR] No backends passed validation. Exiting.")
        return []
    reference = torch.cat(
        [
            a_val[g * m : (g + 1) * m].float() @ b_val[g].float().T
            for g in range(group_size)
        ]
    )
    refchecks = {backend: (lambda o: _cos_sim(reference, o)) for backend in backends}
    problem_flops = 2 * group_size * m * n * k
    problem_bytes = (
        group_size * m * k  # fp8 A
        + group_size * n * k // 2  # fp4 B
        + group_size * m * n * out_dtype.itemsize
    )
    return _bench_and_report(
        args,
        backends,
        run_fns,
        refchecks,
        problem_flops,
        problem_bytes,
        {
            "group_size": group_size,
            "mma_sm": args.mma_sm,
            "input_dtype": "mxfp8",
            "out_dtype": str(out_dtype),
        },
        cos_sim_min=0.97,
    )


# Backend x element-type support of the masked (G, max_m, K) grouped GEMMs.
_MASKED_GROUPED_GEMM_DTYPES = {
    "deepgemm": ["fp8_e4m3"],  # batch_deepgemm_fp8_nt_groupwise, 1x128 / 128x128
    "cute-dsl": ["nvfp4", "mxfp8"],  # grouped_gemm_nt_masked
    "cutile": ["bfloat16", "nvfp4", "mxfp4", "mxfp8"],  # masked_(scaled_)bmm
}

_ELEMENT_BYTES = {
    "bfloat16": 2,
    "fp8_e4m3": 1,
    "mxfp8": 1,
    "nvfp4": 0.5,
    "mxfp4": 0.5,
}


def _make_masked_m(num_groups, expected_m, max_m, seed, device):
    """DeepGEMM-style per-group valid rows (0.7-1.3 x expected_m), rounded up
    to 128 (the cuTile scale-tile granularity) so every backend sees the same
    mask, and clamped to max_m."""
    import random

    rng = random.Random(seed)
    counts = [
        min(max_m, (int(expected_m * rng.uniform(0.7, 1.3)) + 127) // 128 * 128)
        for _ in range(num_groups)
    ]
    return torch.tensor(counts, dtype=torch.int32, device=device)


def _masked_cos_sim(reference, output, masked_m):
    """Cosine similarity over the valid rows of a (G, max_m, N) output."""
    rows = masked_m.tolist()
    return _cos_sim(
        torch.cat([reference[g, : rows[g]] for g in range(len(rows))]),
        torch.cat([output[g, : rows[g]] for g in range(len(rows))]),
    )


def _cutile_block_scaled_operands(a_shape, b_shape, block_scale_type, device):
    """Random block-scaled operands in the cuTile 5D TMA scale layout plus their
    dequantized fp32 values, mirroring tests/gemm/test_{masked,ragged}_scaled_bmm_cutile.py.
    a_shape is (..., M, K) and b_shape (Q, N, K); returns
    (a, b, a_scale, b_scale, a_dequant, b_dequant) with b_dequant (Q, N, K)."""
    from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor

    vec = 16 if block_scale_type == "nvfp4" else 32
    a_ref = MXFP4Tensor(size=a_shape, device=device).random()
    b_ref = MXFP4Tensor(size=b_shape, device=device).random()
    if block_scale_type in ["mxfp8", "mixed"]:
        a_ref = a_ref.to(torch.float32)
        a = a_ref.to(torch.float8_e4m3fn)
    else:
        a = a_ref.to_packed_tensor(dim=len(a_shape) - 1)
    if block_scale_type == "mxfp8":
        b_ref = b_ref.to(torch.float32)
        b = b_ref.to(torch.float8_e4m3fn)
    else:
        b = b_ref.to_packed_tensor(dim=2)

    def make_scale(lead, mn, k):
        shape = [*lead, mn // 128, k // vec // 4, 32, 16]
        raw = torch.rand(shape, device=device) + 1e-8
        if block_scale_type == "nvfp4":
            data = raw.to(torch.float8_e4m3fn)
            ref = data.to(torch.float32)
        else:
            mx = MXScaleTensor(raw)
            data, ref = mx.data, mx.to(torch.float32)
        data = data.reshape(*lead, mn // 128, k // vec // 4, 2, 256)
        # unswizzle (mn_tiles, k_tiles, 32, 4, 4) -> (mn, k) for the reference
        ref = (
            ref.reshape(*lead, mn // 128, k // vec // 4, 32, 4, 4)
            .movedim(-2, -4)
            .movedim(-3, -4)
        )
        ref = ref.reshape(*lead, mn // 128, k // vec // 4, 128, 4)
        ref = ref.movedim(-2, -3).reshape(*lead, mn, k // vec)
        return data, ref.repeat_interleave(vec, dim=-1)

    k = a_shape[-1]
    a_scale, a_scale_ref = make_scale(a_shape[:-2], a_shape[-2], k)
    b_scale, b_scale_ref = make_scale(b_shape[:-2], b_shape[-2], k)
    a_dequant = a_ref.to(torch.float32) * a_scale_ref
    b_dequant = b_ref.to(torch.float32) * b_scale_ref
    return a, b, a_scale, b_scale, a_dequant, b_dequant


def testMaskedGroupedGemm(args):
    """
    Test the masked grouped GEMMs (DeepEP low-latency MoE layout).

    A is (G, max_m, K), B is (G, N, K), and only the first masked_m[g] rows of
    group g are valid. --group_size = G, --m = max_m, --expected_m = mean valid
    rows per group, --input_dtype picks the element type. Backends:
      deepgemm  batch_deepgemm_fp8_nt_groupwise      fp8_e4m3 (1x128 / 128x128 scales)
      cute-dsl  grouped_gemm_nt_masked               nvfp4, mxfp8
      cutile    masked_bmm / masked_scaled_bmm       bfloat16, nvfp4, mxfp4, mxfp8
    Every backend builds its operands in its own scale layout and is checked
    against its own dequantized reference on the valid rows only; all
    backends share the same masked_m.
    """
    if args.verbose >= 1:
        print("[INFO] Running testMaskedGroupedGemm")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")
    warn_if_pdl_unsupported(args, args.routine)
    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    G, max_m, n, k = args.group_size, args.m, args.n, args.k
    dtype = args.input_dtype
    out_dtype = dtype_str_to_torch_dtype(args.out_dtype)
    if out_dtype not in [torch.bfloat16, torch.float16]:
        raise ValueError(f"Unsupported output dtype: {args.out_dtype}")
    if dtype not in _ELEMENT_BYTES:
        raise ValueError(
            f"Unsupported --input_dtype {dtype} for masked_grouped_gemm; choose from {list(_ELEMENT_BYTES)}"
        )
    if max_m % 128 != 0:
        raise ValueError("masked_grouped_gemm requires --m (max_m) % 128 == 0")
    expected_m = args.expected_m if args.expected_m is not None else max_m // 2
    backends = filter_backends_by_compute_capability(
        list(args.backends), args.routine, device
    )
    for backend in list(backends):
        if dtype not in _MASKED_GROUPED_GEMM_DTYPES.get(backend, []):
            print(
                f"[INFO] {backend} does not implement masked_grouped_gemm with input_dtype {dtype}, removing it from backends"
            )
            backends.remove(backend)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return []

    masked_m = _make_masked_m(G, expected_m, max_m, args.random_seed, device)
    run_fns, refchecks = {}, {}

    for backend in list(backends):
        try:
            if backend == "deepgemm":
                a_val = torch.randn(G, max_m, k, device=device)
                b_val = torch.randn(G, n, k, device=device) / np.sqrt(k)
                a_q, a_s = quantize_fp8(a_val, (G, max_m, k // 128), (1, 1, 128), "K")
                b_q, b_s = quantize_fp8(
                    b_val, (G, n // 128, k // 128), (1, 128, 128), "K"
                )
                ref = torch.einsum(
                    "gmk,gnk->gmn",
                    dequantize_fp8(a_q, a_s, "K"),
                    dequantize_fp8(b_q, b_s, "K"),
                )
                a_s, b_s = _deepgemm_packed_scales(
                    a_s, b_s, max_m, n, k, G, sfa_groups=G
                )

                def run_deepgemm(a_q=a_q, b_q=b_q, a_s=a_s, b_s=b_s):
                    return flashinfer.gemm.batch_deepgemm_fp8_nt_groupwise(
                        a_q,
                        b_q,
                        a_s,
                        b_s,
                        masked_m,
                        expected_m,
                        out_dtype=out_dtype,
                        backend="deepgemm",
                    )

                run_fns[backend] = run_deepgemm
            elif backend == "cute-dsl":
                import cutlass
                import cutlass.torch as cutlass_torch
                from flashinfer.cute_dsl.utils import get_cutlass_dtype

                ab_dtype, sf_dtype, sf_vec_size = {
                    "nvfp4": ("float4_e2m1fn", "float8_e4m3fn", 16),
                    "mxfp8": ("float8_e4m3fn", "float8_e8m0fnu", 32),
                }[dtype]
                c_dtype = "bfloat16" if out_dtype == torch.bfloat16 else "float16"
                # Same construction as tests/gemm/test_cute_dsl_blockscaled_gemm.py.
                a_ref = cutlass_torch.matrix(
                    G, max_m, k, False, cutlass.Float32, device=device
                )
                b_ref = cutlass_torch.matrix(
                    G, n, k, False, cutlass.Float32, device=device
                )
                c_ref = cutlass_torch.matrix(
                    G, max_m, n, False, cutlass.Float32, device=device
                )
                _, a_t = cutlass_torch.cute_tensor_like(
                    a_ref,
                    get_cutlass_dtype(ab_dtype),
                    is_dynamic_layout=True,
                    assumed_align=16,
                )
                _, b_t = cutlass_torch.cute_tensor_like(
                    b_ref,
                    get_cutlass_dtype(ab_dtype),
                    is_dynamic_layout=True,
                    assumed_align=16,
                )
                _, c_t = cutlass_torch.cute_tensor_like(
                    c_ref,
                    get_cutlass_dtype(c_dtype),
                    is_dynamic_layout=True,
                    assumed_align=16,
                )
                if ab_dtype == "float4_e2m1fn":
                    # fp4 is passed packed: keep the first half of the byte buffer.
                    a_t = (
                        a_t.permute(2, 0, 1)
                        .flatten()[: a_t.numel() // 2]
                        .reshape(G, max_m, k // 2)
                        .permute(1, 2, 0)
                    )
                    b_t = (
                        b_t.permute(2, 0, 1)
                        .flatten()[: b_t.numel() // 2]
                        .reshape(G, n, k // 2)
                        .permute(1, 2, 0)
                    )
                sfa_ref, _, sfa_t = flashinfer.gemm.create_scale_factor_tensor(
                    G, max_m, k, sf_vec_size, get_cutlass_dtype(sf_dtype), device
                )
                sfb_ref, _, sfb_t = flashinfer.gemm.create_scale_factor_tensor(
                    G, n, k, sf_vec_size, get_cutlass_dtype(sf_dtype), device
                )
                # (m, k, l) x (n, k, l) -> (l, m, n)
                ref = torch.einsum("mkl,nkl->lmn", a_ref * sfa_ref, b_ref * sfb_ref)

                def run_cute_dsl(
                    a_t=a_t,
                    b_t=b_t,
                    c_t=c_t,
                    sfa_t=sfa_t,
                    sfb_t=sfb_t,
                    ab_dtype=ab_dtype,
                    sf_dtype=sf_dtype,
                    c_dtype=c_dtype,
                    sf_vec_size=sf_vec_size,
                ):
                    flashinfer.gemm.grouped_gemm_nt_masked(
                        (a_t, sfa_t),
                        (b_t, sfb_t),
                        c_t,
                        masked_m,
                        ab_dtype=ab_dtype,
                        sf_dtype=sf_dtype,
                        c_dtype=c_dtype,
                        sf_vec_size=sf_vec_size,
                        alpha_dtype="float32",
                    )
                    return c_t.permute(2, 0, 1)  # (m, n, l) -> (l, m, n)

                run_fns[backend] = run_cute_dsl
            elif backend == "cutile" and dtype == "bfloat16":
                a_q = torch.randn(G, max_m, k, device=device, dtype=torch.bfloat16)
                b_q = (torch.randn(G, n, k, device=device) / np.sqrt(k)).to(
                    torch.bfloat16
                )
                ref = torch.einsum("gmk,gnk->gmn", a_q.float(), b_q.float())

                def run_cutile_bf16(a_q=a_q, b_q=b_q):
                    return flashinfer.gemm.masked_bmm(
                        a_q, b_q, masked_m, transpose_b=True, backend="cutile"
                    )

                run_fns[backend] = run_cutile_bf16
            elif backend == "cutile":
                a_q, b_q, a_s, b_s, a_deq, b_deq = _cutile_block_scaled_operands(
                    (G, max_m, k), (G, n, k), dtype, device
                )
                ref = torch.einsum("gmk,gnk->gmn", a_deq, b_deq)

                def run_cutile(a_q=a_q, b_q=b_q, a_s=a_s, b_s=b_s):
                    return flashinfer.gemm.masked_scaled_bmm(
                        a_q,
                        b_q,
                        a_s,
                        b_s,
                        masked_m,
                        dtype,
                        out_dtype=out_dtype,
                        backend="cutile",
                    )

                run_fns[backend] = run_cutile
            refchecks[backend] = lambda o, ref=ref: _masked_cos_sim(ref, o, masked_m)
        except Exception as e:
            print(
                f"[INFO] {backend} backend could not build inputs for this configuration: {type(e).__name__}: {e}"
            )
            backends.remove(backend)

    backends = _drop_failing_backends(backends, lambda bk: run_fns[bk]())
    if len(backends) == 0:
        print("[ERROR] No backends passed validation. Exiting.")
        return []

    valid_m = int(masked_m.sum().item())
    elem = _ELEMENT_BYTES[dtype]
    problem_flops = 2 * valid_m * n * k
    problem_bytes = int(
        valid_m * k * elem + G * n * k * elem + valid_m * n * out_dtype.itemsize
    )
    return _bench_and_report(
        args,
        backends,
        run_fns,
        refchecks,
        problem_flops,
        problem_bytes,
        {
            "group_size": G,
            "expected_m": expected_m,
            "input_dtype": dtype,
            "out_dtype": str(out_dtype),
        },
        cos_sim_min=0.97 if "fp4" in dtype else 0.99,
        # DeepGEMM reloads its cubin on every call (deep_gemm.py load() cache
        # key mismatch), which CUDA-graph capture cannot record.
        no_graph_backends=("deepgemm",),
    )


# Backend x element-type support of the ragged (m_indptr) grouped GEMMs.
_GROUPED_MM_DTYPES = {
    "cudnn": ["bfloat16", "fp8_e4m3", "mxfp8", "nvfp4"],  # grouped_mm_*
    "cutile": ["bfloat16", "mxfp8", "nvfp4", "mxfp4"],  # ragged_(scaled_)bmm
}


def testGroupedMm(args):
    """
    Test the ragged grouped GEMMs (m_indptr segments, no groupwise scales).

    group_size uniform segments of m rows; B is (group_size, N, K). Backends:
      cudnn   grouped_mm_bf16 / _fp8 (per-tensor) / _mxfp8 / _fp4 (nvfp4)
      cutile  ragged_bmm (bf16) / ragged_scaled_bmm (mxfp8, nvfp4, mxfp4)
    cudnn is checked against the unquantized bf16 product; cuTile
    block-scaled operands are random values in its 5D scale layout and are
    checked against their own dequantized reference.
    """
    if args.verbose >= 1:
        print("[INFO] Running testGroupedMm")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")
    warn_if_pdl_unsupported(args, args.routine)
    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    G, m, n, k = args.group_size, args.m, args.n, args.k
    dtype = args.input_dtype
    out_dtype = dtype_str_to_torch_dtype(args.out_dtype)
    if dtype not in _ELEMENT_BYTES:
        raise ValueError(
            f"Unsupported --input_dtype {dtype} for grouped_mm; choose from {list(_ELEMENT_BYTES)}"
        )
    backends = filter_backends_by_compute_capability(
        list(args.backends), args.routine, device
    )
    for backend in list(backends):
        if dtype not in _GROUPED_MM_DTYPES.get(backend, []):
            print(
                f"[INFO] {backend} does not implement grouped_mm with input_dtype {dtype}, removing it from backends"
            )
            backends.remove(backend)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return []

    total_m = G * m
    m_indptr = torch.arange(0, G + 1, dtype=torch.int32, device=device) * m
    a_bf16 = torch.randn(total_m, k, device=device, dtype=torch.bfloat16)
    b_bf16 = (torch.randn(G, n, k, device=device) / np.sqrt(k)).to(torch.bfloat16)
    bf16_reference = torch.cat(
        [a_bf16[g * m : (g + 1) * m].float() @ b_bf16[g].float().T for g in range(G)]
    )
    run_fns, refchecks = {}, {}

    for backend in list(backends):
        try:
            ref = bf16_reference
            if backend == "cudnn" and dtype == "bfloat16":

                def run_bf16():
                    return flashinfer.grouped_mm_bf16(
                        a_bf16, b_bf16, m_indptr, out_dtype=out_dtype
                    )

                run_fns[backend] = run_bf16
            elif backend == "cudnn" and dtype == "fp8_e4m3":
                a_q, a_inv = to_float8(a_bf16)
                b_q, b_inv = to_float8(b_bf16)
                alpha = (a_inv * b_inv).float().reshape(1)

                def run_fp8(a_q=a_q, b_q=b_q, alpha=alpha):
                    return flashinfer.grouped_mm_fp8(
                        a_q, b_q, m_indptr, alpha=alpha, out_dtype=out_dtype
                    )

                run_fns[backend] = run_fp8
            elif backend == "cudnn" and dtype == "mxfp8":
                a_q, a_s = mxfp8_quantize(a_bf16, is_sf_swizzled_layout=True)
                b_q, b_s = mxfp8_quantize(b_bf16, is_sf_swizzled_layout=True)
                a_s = a_s.reshape(-1, k // 32)
                b_s = b_s.reshape(G, -1, k // 32)

                def run_mxfp8(a_q=a_q, b_q=b_q, a_s=a_s, b_s=b_s):
                    return flashinfer.grouped_mm_mxfp8(
                        a_q, b_q, a_s, b_s, m_indptr, out_dtype=out_dtype
                    )

                run_fns[backend] = run_mxfp8
            elif backend == "cudnn" and dtype == "nvfp4":
                a_gsf = (448 * 6) / a_bf16.float().abs().nan_to_num().max()
                b_gsf = (448 * 6) / b_bf16.float().abs().nan_to_num().max()
                a_q, a_s = flashinfer.nvfp4_quantize(
                    a_bf16,
                    a_gsf,
                    sfLayout=flashinfer.SfLayout.layout_128x4,
                    do_shuffle=False,
                )
                b_q, b_s = flashinfer.nvfp4_quantize(
                    b_bf16.reshape(G * n, k),
                    b_gsf,
                    sfLayout=flashinfer.SfLayout.layout_128x4,
                    do_shuffle=False,
                )
                a_s = a_s.view(torch.float8_e4m3fn).reshape(-1, k // 16)
                b_q = b_q.reshape(G, n, k // 2)
                b_s = b_s.view(torch.float8_e4m3fn).reshape(G, -1, k // 16)
                alpha = torch.tensor(
                    [1.0 / (a_gsf * b_gsf)], dtype=torch.float32, device=device
                )

                def run_fp4(a_q=a_q, b_q=b_q, a_s=a_s, b_s=b_s, alpha=alpha):
                    return flashinfer.grouped_mm_fp4(
                        a_q,
                        b_q,
                        a_s,
                        b_s,
                        m_indptr,
                        alpha=alpha,
                        out_dtype=out_dtype,
                        block_size=16,
                    )

                run_fns[backend] = run_fp4
            elif backend == "cutile" and dtype == "bfloat16":

                def run_ragged_bf16():
                    return flashinfer.gemm.ragged_bmm(
                        a_bf16,
                        b_bf16,
                        m_indptr,
                        m,
                        None,
                        transpose_b=True,
                        out_dtype=out_dtype,
                        backend="cutile",
                    )

                run_fns[backend] = run_ragged_bf16
            elif backend == "cutile":
                if m % 128 != 0:
                    raise ValueError(
                        "ragged_scaled_bmm needs 128-row aligned segments (m % 128 == 0)"
                    )
                a_q, b_q, a_s, b_s, a_deq, b_deq = _cutile_block_scaled_operands(
                    (total_m, k), (G, n, k), dtype, device
                )
                ref = torch.cat(
                    [a_deq[g * m : (g + 1) * m] @ b_deq[g].T for g in range(G)]
                )

                def run_ragged_scaled(a_q=a_q, b_q=b_q, a_s=a_s, b_s=b_s):
                    return flashinfer.gemm.ragged_scaled_bmm(
                        a_q, b_q, a_s, b_s, m_indptr, m, dtype, backend="cutile"
                    )

                run_fns[backend] = run_ragged_scaled
            refchecks[backend] = lambda o, ref=ref: _cos_sim(ref, o)
        except Exception as e:
            print(
                f"[INFO] {backend} backend could not build inputs for this configuration: {type(e).__name__}: {e}"
            )
            backends.remove(backend)

    backends = _drop_failing_backends(backends, lambda bk: run_fns[bk]())
    if len(backends) == 0:
        print("[ERROR] No backends passed validation. Exiting.")
        return []

    elem = _ELEMENT_BYTES[dtype]
    problem_flops = 2 * total_m * n * k
    problem_bytes = int(
        total_m * k * elem + G * n * k * elem + total_m * n * out_dtype.itemsize
    )
    return _bench_and_report(
        args,
        backends,
        run_fns,
        refchecks,
        problem_flops,
        problem_bytes,
        {"group_size": G, "input_dtype": dtype, "out_dtype": str(out_dtype)},
        cos_sim_min=0.97 if dtype != "bfloat16" else 0.99,
    )


def testMmNvfp4Svdquant(args):
    """
    Test mm_nvfp4_svdquant: out = alpha * (a @ b^T + d @ l1^T) + bias.

    a/b are NVFP4 (128x4 scales) of a smoothed activation and a weight; d is
    the rank-r LoRA-down output and l1 the LoRA-up weight pre-divided by alpha
    (operand construction follows benchmarks/bench_nvfp4_svdquant_gemm.py).
    --rank sets r (multiple of 32), --bias adds the bias epilogue. Backends:
    cutlass (SM100/103/107), cute-dsl / cute-dsl-unfused (SM120/121),
    cutlass-sm120 (SM120), auto.
    Refcheck: cosine similarity against the bf16 smoothed GEMM + LoRA term.
    """
    if args.verbose >= 1:
        print("[INFO] Running testMmNvfp4Svdquant")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")
    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    m, n, k, rank = args.m, args.n, args.k, args.rank
    x = torch.randn(m, k, dtype=torch.bfloat16, device=device) / (k**0.25)
    pqs = (1.0 + 0.3 * torch.randn(k, dtype=torch.bfloat16, device=device)).abs()
    smoothed = (x * pqs).to(torch.bfloat16)
    x_gsf = ((448.0 * 6.0) / smoothed.float().abs().nan_to_num().max()).reshape(1)
    w = torch.randn(n, k, dtype=torch.bfloat16, device=device) / (k**0.25)
    w_gsf = ((448.0 * 6.0) / w.float().abs().nan_to_num().max()).reshape(1)
    xq, x_sf = flashinfer.nvfp4_quantize(
        smoothed, x_gsf, sfLayout=flashinfer.SfLayout.layout_128x4, do_shuffle=False
    )
    wq, w_sf = flashinfer.nvfp4_quantize(
        w, w_gsf, sfLayout=flashinfer.SfLayout.layout_128x4, do_shuffle=False
    )
    xq, wq = xq.view(torch.uint8), wq.view(torch.uint8)
    x_sf = x_sf.view(torch.uint8).reshape(-1).contiguous()
    w_sf = w_sf.view(torch.uint8).reshape(-1).contiguous()
    alpha = (1.0 / (x_gsf * w_gsf)).reshape(1).float()
    lora_a = torch.randn(rank, k, dtype=torch.bfloat16, device=device) / (k**0.25)
    lora_b = torch.randn(n, rank, dtype=torch.bfloat16, device=device) / (rank**0.25)
    d = torch.mm(x, (pqs.unsqueeze(1) * lora_a.t()).to(torch.bfloat16))
    l1 = (lora_b.float() / alpha).to(torch.bfloat16).contiguous()
    bias = torch.randn(n, dtype=torch.bfloat16, device=device) if args.bias else None
    out = torch.empty(m, n, dtype=torch.bfloat16, device=device)

    def make_run(backend):
        def run():
            return flashinfer.mm_nvfp4_svdquant(
                xq,
                wq,
                x_sf,
                w_sf,
                alpha,
                d,
                l1,
                bias=bias,
                out=out,
                backend=backend,
                enable_pdl=args.enable_pdl,
            )

        return run

    run_fns = {
        backend: make_run(backend)
        for backend in [
            "cutlass",
            "cute-dsl",
            "cute-dsl-unfused",
            "cutlass-sm120",
            "auto",
        ]
    }
    backends = _filter_and_probe(args, device, list(args.backends), run_fns)
    if len(backends) == 0:
        print("[ERROR] No backends passed validation. Exiting.")
        return []
    reference = smoothed.float() @ w.float().T + d.float() @ lora_b.float().T
    if bias is not None:
        reference = reference + bias.float()
    refchecks = {backend: (lambda o: _cos_sim(reference, o)) for backend in backends}
    problem_flops = 2 * m * n * k + 2 * m * n * rank
    problem_bytes = (
        (m * k + n * k) // 2  # nvfp4 a, b
        + (m * rank + n * rank) * 2  # bf16 d, l1
        + m * n * 2  # bf16 out
    )
    return _bench_and_report(
        args,
        backends,
        run_fns,
        refchecks,
        problem_flops,
        problem_bytes,
        {
            "rank": rank,
            "bias": args.bias,
            "input_dtype": "nvfp4",
            "out_dtype": str(torch.bfloat16),
            "enable_pdl": args.enable_pdl,
        },
        cos_sim_min=0.97,
        autotune_backends=(
            "cutlass",
            "cute-dsl",
            "cute-dsl-unfused",
            "cutlass-sm120",
            "auto",
        ),
    )


_FUSED_LINEAR_HEAD_DIM = 128  # the QKV epilogue supports head_dim 128 only


def testFusedLinear(args):
    """
    Test the PrimsTS dense GEMMs with fused epilogues (flashinfer.gemm):

      --epilogue linear            fp8_linear / fp4_linear
      --epilogue swiglu            fp8_linear_swiglu / fp4_linear_swiglu
                                   (weight rows are interleaved gate/up pairs;
                                   output width N / 2)
      --epilogue qkv_qknorm_rope   fp8_qkv_qknorm_rope / fp4_qkv_qknorm_rope
                                   (N = 3 * heads * 128; per-head RMSNorm of Q
                                   and K, then interleaved RoPE)

    --input_dtype fp8_e4m3 uses per-token / per-channel fp32 scales; nvfp4
    uses 128x4 block scales from flashinfer.nvfp4_quantize plus host-float
    global scales. Output is bfloat16. --bias applies to linear / swiglu.
    Refcheck: cosine similarity against an fp32 dequantized reference.
    """
    if args.verbose >= 1:
        print("[INFO] Running testFusedLinear")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")
    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    m, n, k = args.m, args.n, args.k
    epilogue = args.epilogue
    dtype = args.input_dtype
    if dtype not in ["fp8_e4m3", "nvfp4"]:
        raise ValueError("fused_linear supports --input_dtype fp8_e4m3 or nvfp4")
    if dtype_str_to_torch_dtype(args.out_dtype) != torch.bfloat16:
        raise ValueError("fused_linear benchmarks bfloat16 output only")
    if epilogue == "qkv_qknorm_rope" and args.bias:
        raise ValueError("the qkv_qknorm_rope epilogue has no bias")
    num_heads = None
    if epilogue == "qkv_qknorm_rope":
        if n % (3 * _FUSED_LINEAR_HEAD_DIM) != 0:
            raise ValueError("qkv_qknorm_rope needs n % 384 == 0 (head_dim 128)")
        num_heads = n // (3 * _FUSED_LINEAR_HEAD_DIM)
    out_n = n // 2 if epilogue == "swiglu" else n

    a_val = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    w_val = (torch.randn(n, k, device=device) / np.sqrt(k)).to(torch.bfloat16)
    bias = (
        torch.randn(n, device=device, dtype=torch.bfloat16) * 0.1 if args.bias else None
    )
    if dtype == "fp8_e4m3":
        # Per-token / per-channel scales (amax / 448), as serving stacks use.
        a_scale = (a_val.float().abs().amax(dim=1) / 448.0).clamp(min=1e-12)
        w_scale = (w_val.float().abs().amax(dim=1) / 448.0).clamp(min=1e-12)
        a_q = (a_val.float() / a_scale[:, None]).to(torch.float8_e4m3fn)
        w_q = (w_val.float() / w_scale[:, None]).to(torch.float8_e4m3fn)
        a_scale, w_scale = a_scale.contiguous(), w_scale.contiguous()
        linear_ref = (a_q.float() * a_scale[:, None]) @ (
            w_q.float() * w_scale[:, None]
        ).T
        operands = (a_q, w_q, a_scale, w_scale)
        in_bytes = m * k + n * k
    else:
        a_gsf = (448 * 6) / a_val.float().abs().nan_to_num().max()
        w_gsf = (448 * 6) / w_val.float().abs().nan_to_num().max()
        a_q, a_sf = flashinfer.nvfp4_quantize(
            a_val, a_gsf, sfLayout=flashinfer.SfLayout.layout_128x4, do_shuffle=False
        )
        w_q, w_sf = flashinfer.nvfp4_quantize(
            w_val, w_gsf, sfLayout=flashinfer.SfLayout.layout_128x4, do_shuffle=False
        )
        operands = (
            a_q.view(torch.uint8),
            a_sf.reshape(-1).contiguous(),
            1.0 / float(a_gsf),
            w_q.view(torch.uint8),
            w_sf.reshape(-1).contiguous(),
            1.0 / float(w_gsf),
        )
        # nvfp4 quantization error is the refcheck's tolerance; compare with
        # the unquantized product.
        linear_ref = a_val.float() @ w_val.float().T
        in_bytes = (m * k + n * k) // 2
    if bias is not None:
        linear_ref = linear_ref + bias.float()

    rope_args = {}
    if epilogue == "qkv_qknorm_rope":
        hd = _FUSED_LINEAR_HEAD_DIM
        q_norm = torch.rand(hd, device=device, dtype=torch.bfloat16)
        k_norm = torch.rand(hd, device=device, dtype=torch.bfloat16)
        angles = torch.randn(m, hd // 2, device=device)
        cos_sin = torch.cat((angles.cos(), angles.sin()), dim=-1).contiguous()
        positions = torch.arange(m, device=device, dtype=torch.int64)
        rope_args = dict(
            q_norm=q_norm, k_norm=k_norm, cos_sin=cos_sin, positions=positions
        )

    def run():
        gemm = flashinfer.gemm
        if epilogue == "qkv_qknorm_rope":
            fn = (
                gemm.fp8_qkv_qknorm_rope
                if dtype == "fp8_e4m3"
                else gemm.fp4_qkv_qknorm_rope
            )
            return fn(
                *operands,
                rope_args["q_norm"],
                rope_args["k_norm"],
                rope_args["cos_sin"],
                rope_args["positions"],
                num_q_heads=num_heads,
                num_kv_heads=num_heads,
                head_dim=_FUSED_LINEAR_HEAD_DIM,
            )
        if dtype == "fp8_e4m3":
            fn = gemm.fp8_linear_swiglu if epilogue == "swiglu" else gemm.fp8_linear
        else:
            fn = gemm.fp4_linear_swiglu if epilogue == "swiglu" else gemm.fp4_linear
        return fn(*operands, bias)

    run_fns = {"prims-ts": run}
    backends = _filter_and_probe(args, device, list(args.backends), run_fns)
    if len(backends) == 0:
        print("[ERROR] No backends passed validation. Exiting.")
        return []

    if epilogue == "linear":
        reference = linear_ref
    elif epilogue == "swiglu":
        reference = linear_ref[:, 0::2] * F.silu(linear_ref[:, 1::2])
    else:
        hd, half = _FUSED_LINEAR_HEAD_DIM, _FUSED_LINEAR_HEAD_DIM // 2
        shaped = linear_ref.bfloat16().float().view(m, 3, num_heads, hd).clone()
        cos_sin = rope_args["cos_sin"]
        cos, sin = cos_sin[:, None, :half], cos_sin[:, None, half:]
        for part, norm_weight in ((0, rope_args["q_norm"]), (1, rope_args["k_norm"])):
            value = shaped[:, part]
            value = (
                value
                * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
                * norm_weight.float()
            )
            pairs = value.view(m, num_heads, half, 2)
            shaped[:, part] = torch.stack(
                (
                    pairs[..., 0] * cos - pairs[..., 1] * sin,
                    pairs[..., 1] * cos + pairs[..., 0] * sin,
                ),
                -1,
            ).flatten(-2)
        reference = shaped.reshape(m, n)
    refchecks = {"prims-ts": lambda o: _cos_sim(reference, o)}
    problem_flops = 2 * m * n * k
    problem_bytes = in_bytes + m * out_n * torch.bfloat16.itemsize
    extra = {
        "epilogue": epilogue,
        "input_dtype": dtype,
        "out_dtype": str(torch.bfloat16),
        "bias": args.bias,
    }
    if num_heads is not None:
        extra["num_qo_heads"] = num_heads
        extra["num_kv_heads"] = num_heads
    return _bench_and_report(
        args,
        backends,
        run_fns,
        refchecks,
        problem_flops,
        problem_bytes,
        extra,
        cos_sim_min=0.97 if dtype == "nvfp4" else 0.99,
        autotune_backends=("prims-ts",),
    )
