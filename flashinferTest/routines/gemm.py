import argparse
from collections import defaultdict

import numpy as np
import torch
from einops import einsum

import flashinfer
from flashinfer.testing.utils import (
    bench_gpu_time,
    bench_gpu_time_with_cudagraph,
    dequantize_fp8,
    quantize_fp8,
    set_seed,
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
        "--out_dtype",
        type=str,
        required=False,
        default="bfloat16",
        choices=["bfloat16", "float16"],
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

    args = parser.parse_args(line)
    if args.verbose >= 1:
        print(f"[INFO] {args = }")
    return args


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
    if args.verbose >= 1:
        print(f"[INFO] Running testGemmFp8NtGroupwise")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    # Basic setup
    set_seed(args.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device()).replace(" ", "_")
    if args.verbose >= 2:
        print(f"[VVERBOSE] {gpu_name = }")

    m = args.m
    n = args.n
    k = args.k
    group_size = args.group_size  # Unused for gemm_fp8_nt_groupwise
    tile_size = args.tile_size
    scale_major_mode = args.scale_major_mode
    mma_sm = args.mma_sm
    is_cuda_graph_compatible = not args.no_cuda_graph
    run_refcheck = args.refcheck

    if args.out_dtype == "bfloat16":
        out_dtype = torch.bfloat16
    elif args.out_dtype == "float16":
        out_dtype = torch.float16
    else:
        raise ValueError(f"Unsupported output dtype: {args.out_dtype}")

    a_val = torch.randn((m, k), dtype=torch.float, device=device)
    b_val = torch.randn((n, k), dtype=torch.float, device=device) / np.sqrt(k)

    if args.verbose >= 2:
        print(f"[VVERBOSE] {a_val.shape = }")
        print(f"[VVERBOSE] {b_val.shape = }")

    if scale_major_mode == "K":
        a_scale_shape = (m, k // tile_size)
        b_scale_shape = (n // tile_size, k // tile_size)
    else:
        a_scale_shape = (k // tile_size, m)
        b_scale_shape = (k // tile_size, n // tile_size)

    a_tile_shape = (1, tile_size)
    b_tile_shape = (tile_size, tile_size)

    a_fp8, a_scale = quantize_fp8(a_val, a_scale_shape, a_tile_shape, scale_major_mode)
    b_fp8, b_scale = quantize_fp8(b_val, b_scale_shape, b_tile_shape, scale_major_mode)

    if args.verbose >= 2:
        print(f"[VVERBOSE] {a_fp8.shape = }")
        print(f"[VVERBOSE] {b_fp8.shape = }")
        print(f"[VVERBOSE] {a_scale.shape = }")
        print(f"[VVERBOSE] {b_scale.shape = }")

    a_dequant = dequantize_fp8(a_fp8, a_scale, scale_major_mode)
    b_dequant = dequantize_fp8(b_fp8, b_scale, scale_major_mode)

    kernel_fn = lambda: flashinfer.gemm.gemm_fp8_nt_groupwise(
        a=a_fp8,
        b=b_fp8,
        a_scale=a_scale,
        b_scale=b_scale,
        scale_major_mode=scale_major_mode,
        out_dtype=out_dtype,
        mma_sm=mma_sm,
    )

    c = kernel_fn()

    if run_refcheck:
        ref_c = einsum(a_dequant, b_dequant, "m k, n k -> m n").to(out_dtype)
        try:
            torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
        except AssertionError as e:
            print(
                f"[ERROR] Output mismatch between reference (torch.matmul) and gemm_fp8_nt_groupwise."
            )
            if not args.allow_output_mismatch:
                print(e)
                raise

    if is_cuda_graph_compatible:
        measured_times = bench_gpu_time_with_cudagraph(
            fn=kernel_fn,
            dry_runs=args.dry_run_iters,
            num_iters=args.num_iters,
            nvtx_range_name=f"gemm_fp8_nt_groupwise",
            l2_flush=True,
            l2_flush_size_mb=256,
            l2_flush_device=device,
            sleep_after_run=True,  # GEMMs are very MMA-heavy, so prefer sleep to reduce throttling.
        )
    else:
        measured_times = bench_gpu_time(
            fn=kernel_fn,
            dry_runs=args.dry_run_iters,
            num_iters=args.num_iters,
            nvtx_range_name=f"gemm_fp8_nt_groupwise",
            l2_flush=True,
            l2_flush_size_mb=256,
            l2_flush_device=device,
            sleep_after_run=True,  # GEMMs are very MMA-heavy, so prefer sleep to reduce throttling.
        )

    median_time = np.median(measured_times)  # in msec
    std_time = np.std(measured_times)  # in msec
    problem_flops = 2 * m * n * k
    problem_bytes = (m * k + n * k) * torch.float8_e4m3fn.itemsize + (
        m * n
    ) * out_dtype.itemsize
    tflops = problem_flops / (10**9 * median_time)  # in TFLOPs/sec
    tb_per_sec = problem_bytes / (10**9 * median_time)  # in TB/sec

    print(
        f"[PERF] {'gemm_fp8_nt_groupwise'.ljust(8)[:8]}:: median time {median_time:.3f} ms; std {std_time:.3f} ms; achieved tflops {tflops:.3f} TFLOPs/sec; achieved tb_per_sec {tb_per_sec:.3f} TB/sec"
    )

    res = []
    if args.output_path is not None:
        cur_res = defaultdict(str)
        cur_res["routine"] = "gemm_fp8_nt_groupwise"
        cur_res["median_time"] = median_time
        cur_res["std_time"] = std_time
        cur_res["tflops"] = tflops
        cur_res["tb_per_sec"] = tb_per_sec
        cur_res["m"] = m
        cur_res["n"] = n
        cur_res["k"] = k
        cur_res["tile_size"] = tile_size
        cur_res["group_size"] = group_size
        cur_res["scale_major_mode"] = scale_major_mode
        cur_res["out_dtype"] = out_dtype
        cur_res["mma_sm"] = mma_sm
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
    if args.verbose >= 1:
        print(f"[INFO] Running testGroupGemmFp8NtGroupwise")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    # Basic setup
    set_seed(args.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device()).replace(" ", "_")
    if args.verbose >= 2:
        print(f"[VVERBOSE] {gpu_name = }")

    m = args.m
    n = args.n
    k = args.k
    group_size = args.group_size
    tile_size = args.tile_size
    scale_major_mode = args.scale_major_mode
    mma_sm = args.mma_sm
    is_cuda_graph_compatible = not args.no_cuda_graph
    run_refcheck = args.refcheck

    if args.out_dtype == "bfloat16":
        out_dtype = torch.bfloat16
    elif args.out_dtype == "float16":
        out_dtype = torch.float16
    else:
        raise ValueError(f"Unsupported output dtype: {args.out_dtype}")

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

    if args.verbose >= 2:
        print(f"[VVERBOSE] {a_fp8.shape = }")
        print(f"[VVERBOSE] {b_fp8.shape = }")
        print(f"[VVERBOSE] {a_scale.shape = }")
        print(f"[VVERBOSE] {b_scale.shape = }")
        print(f"[VVERBOSE] {m_indptr.shape = }")

    kernel_fn = lambda: flashinfer.gemm.group_gemm_fp8_nt_groupwise(
        a=a_fp8,
        b=b_fp8,
        a_scale=a_scale,
        b_scale=b_scale,
        m_indptr=m_indptr,
        scale_major_mode=scale_major_mode,
        out_dtype=out_dtype,
        mma_sm=mma_sm,
    )

    c = kernel_fn()

    print(f"[INFO] {a_dequant.shape = }")
    print(f"[INFO] {b_dequant.shape = }")
    print(f"[INFO] {c.shape = }")

    if run_refcheck:
        ref_c = (
            einsum(
                a_dequant.view((group_size, m, k)),
                b_dequant,
                "b m k, b n k -> b m n",
            )
            .view((group_size * m, n))
            .to(out_dtype)
        )
        try:
            torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
        except AssertionError as e:
            print(
                f"[ERROR] Output mismatch between reference (torch.matmul) and group_gemm_fp8_nt_groupwise."
            )
            if not args.allow_output_mismatch:
                print(e)
                raise

    if is_cuda_graph_compatible:
        measured_times = bench_gpu_time_with_cudagraph(
            fn=kernel_fn,
            dry_runs=args.dry_run_iters,
            num_iters=args.num_iters,
            nvtx_range_name=f"group_gemm_fp8_nt_groupwise",
            l2_flush=True,
            l2_flush_size_mb=256,
            l2_flush_device=device,
            sleep_after_run=True,  # GEMMs are very MMA-heavy, so prefer sleep to reduce throttling.
        )
    else:
        measured_times = bench_gpu_time(
            fn=kernel_fn,
            dry_runs=args.dry_run_iters,
            num_iters=args.num_iters,
            nvtx_range_name=f"group_gemm_fp8_nt_groupwise",
            l2_flush=True,
            l2_flush_size_mb=256,
            l2_flush_device=device,
            sleep_after_run=True,  # GEMMs are very MMA-heavy, so prefer sleep to reduce throttling.
        )

    median_time = np.median(measured_times)  # in msec
    std_time = np.std(measured_times)  # in msec
    problem_flops = 2 * m * n * k
    problem_bytes = (m * k + n * k) * torch.float8_e4m3fn.itemsize + (
        m * n
    ) * out_dtype.itemsize
    tflops = problem_flops / (10**9 * median_time)  # in TFLOPs/sec
    tb_per_sec = problem_bytes / (10**9 * median_time)  # in TB/sec

    print(
        f"[PERF] {'grp_gemm_fp8_nt_groupwise'.ljust(8)[:8]}:: median time {median_time:.3f} ms; std {std_time:.3f} ms; achieved tflops {tflops:.3f} TFLOPs/sec; achieved tb_per_sec {tb_per_sec:.3f} TB/sec"
    )

    res = []
    if args.output_path is not None:
        cur_res = defaultdict(str)
        cur_res["routine"] = "group_gemm_fp8_nt_groupwise"
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
        res.append(cur_res)
    return res
