import torch
from flashinfer import (
    SfLayout,
    autotune,
    mm_fp4,
    nvfp4_quantize,
    mxfp4_quantize,
)
from flashinfer.jit.core import logger
from flashinfer.testing.utils import bench_gpu_time
from flashinfer.utils import get_compute_capability

import argparse
from functools import partial
from itertools import product
import logging
import numpy as np
from typing import Literal


def _bench_mm_fp4(
    m: int,
    n: int,
    k: int,
    res_dtype: torch.dtype,
    backend: Literal["cudnn", "trtllm", "cutlass", "auto"],
    use_128x4_sf_layout: bool,
    fp4_type: str,
    do_autotune: bool = False,
    warmups: int = 100,
    iterations: int = 100,
) -> tuple[float, float]:
    use_nvfp4 = fp4_type == "nvfp4"

    compute_capability = get_compute_capability(torch.device(device="cuda"))
    compute_capability_number = compute_capability[0] * 10 + compute_capability[1]
    if not mm_fp4.is_backend_supported(backend, compute_capability_number):
        print(
            f"Skipping test for {backend} because it is not supported on compute capability {compute_capability_number}."
        )
        return

    if backend == "trtllm":
        if res_dtype == torch.float16:
            print("Skipping test for trtllm fp4 with float16")
            return
        if compute_capability[0] in [11, 12]:
            print("trtllm gemm does not support SM110/SM120/SM121 GPUs.")
            return
    if not use_128x4_sf_layout and backend != "trtllm":
        print("Skipping test for non-trtllm fp4 with use_128x4_sf_layout=False")
        return
    if not use_nvfp4 and backend not in ["cudnn", "auto"]:
        print("mx_fp4 is only supported for cudnn and auto backends")
        return

    input = torch.randn([m, k], device="cuda", dtype=torch.bfloat16)
    mat2 = torch.randn([n, k], device="cuda", dtype=torch.bfloat16)
    a_sf_layout = SfLayout.layout_128x4 if use_128x4_sf_layout else SfLayout.layout_8x4

    global_sf_input = (448 * 6) / input.float().abs().nan_to_num().max()
    global_sf_mat2 = (448 * 6) / mat2.float().abs().nan_to_num().max()

    # for trtllm, we need to shuffle mat2 because we swap A, B.
    do_shuffle_b = backend == "trtllm"

    block_size = 16 if use_nvfp4 else 32
    has_alpha = fp4_type == "mxfp4_alpha" or fp4_type == "nvfp4"

    if use_nvfp4:
        input_fp4, input_inv_s = nvfp4_quantize(
            input, global_sf_input, sfLayout=a_sf_layout, do_shuffle=False
        )
        mat2_fp4, mat2_inv_s = nvfp4_quantize(
            mat2,
            global_sf_mat2,
            sfLayout=SfLayout.layout_128x4,
            do_shuffle=do_shuffle_b,
        )
    else:
        input_fp4, input_inv_s = mxfp4_quantize(input)
        mat2_fp4, mat2_inv_s = mxfp4_quantize(mat2)

    alpha = 1.0 / (global_sf_input * global_sf_mat2) if has_alpha else None

    res = torch.empty([m, n], device="cuda", dtype=res_dtype)

    fn = partial(
        mm_fp4,
        alpha=alpha,
        out_dtype=res_dtype,
        out=res,
        block_size=block_size,
        use_8x4_sf_layout=not use_128x4_sf_layout,
        backend=backend,
        use_nvfp4=use_nvfp4,
    )

    input_kwargs = {
        "a": input_fp4,
        "b": mat2_fp4.T,
        "a_descale": input_inv_s,
        "b_descale": mat2_inv_s.T,
    }
    try:

        def bench(do_autotune: bool) -> float:
            with autotune(do_autotune):
                fn(**input_kwargs)
            ms_list = bench_gpu_time(
                fn,
                dry_run_iters=warmups,
                repeat_iters=iterations,
                enable_cupti=False,  # FIXME: Cupti causes CUDA Illegal Memory Access
                use_cuda_graph=True,
                input_kwargs=input_kwargs,
                cold_l2_cache=True,
            )
            median_ms = np.median(ms_list)
            return median_ms

        ms = bench(do_autotune=do_autotune)
        tflops = 2 * m * n * k * 1e-9 / ms
        return ms, tflops
    except Exception as e:
        print(f"    Error: {e}")
        return 0, 0


if __name__ == "__main__":
    # suppress autotuner's logs
    logger.setLevel(logging.WARNING)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--m", type=int, nargs="+", default=[1, 64, 128, 1024, 2048, 4096, 8192]
    )
    parser.add_argument("--n", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    parser.add_argument(
        "--k", type=int, nargs="+", default=[1024, 2048, 4096, 8192, 16384, 32768]
    )
    parser.add_argument(
        "--backend", type=str, nargs="+", default=["cudnn", "trtllm", "cutlass"]
    )
    args = parser.parse_args()

    for m, n, k in product(args.m, args.n, args.k):
        print(f"m={m}, n={n}, k={k}".center(100, "-"))
        for backend in args.backend:
            print(f"  {backend}:")
            ms, tflops = _bench_mm_fp4(
                m, n, k, torch.bfloat16, backend, True, "nvfp4", False
            )
            us = ms * 1e3
            print(
                f"    w/o autotune: {f'{us:.3f} us' if us < 1000 else f'{us / 1000:.3f} ms'}, {tflops:.2f} TFLOPs/s"
            )
            ms, tflops = _bench_mm_fp4(
                m, n, k, torch.bfloat16, backend, True, "nvfp4", True
            )
            us = ms * 1e3
            print(
                f"    with autotune: {f'{us:.3f} us' if us < 1000 else f'{us / 1000:.3f} ms'}, {tflops:.2f} TFLOPs/s"
            )
