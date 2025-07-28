import numpy as np
import torch
import triton

import flashinfer
import flashinfer.triton
from flashinfer.testing.utils import bench_gpu_time


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def supports_tma():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


def bench_gemm_persistent(num_sms, dtype, M, N, K, reps=1000, warmup_reps=10000):
    measurements = bench_gpu_time(
        lambda: flashinfer.triton.sm_constraint_gemm.gemm_persistent(
            a=torch.randn((M, K), device="cuda", dtype=torch.float16).to(dtype),
            b=torch.randn((N, K), device="cuda", dtype=torch.float16).to(dtype),
            alpha=1.0,
            beta=0.0,
            num_sms=num_sms,
        ),
        dry_runs=warmup_reps,
        num_iters=reps,
        l2_flush=True,
        l2_flush_size_mb=256,
        l2_flush_device=torch.device("cuda:0"),
    )
    ms = np.median(measurements)

    # matmul: 2 * M * N * K
    # scale and add: 3 * M * N
    flops = (2 * M * N * K + 3 * M * N) / ms / 1e9
    print(
        f"GEMM Persistent | num_sms: {num_sms}, M: {M}, N: {N}, K: {K}, {dtype}: {flops:.3f} TFLOPs/s"
    )


def bench_gemm_descriptor_persistent(
    num_sms, dtype, M, N, K, reps=1000, warmup_reps=10000
):
    if dtype == torch.float32:
        return
    measurements = bench_gpu_time(
        lambda: flashinfer.triton.sm_constraint_gemm.gemm_descriptor_persistent(
            a=torch.randn((M, K), device="cuda", dtype=torch.float16).to(dtype),
            b=torch.randn((N, K), device="cuda", dtype=torch.float16).to(dtype),
            alpha=1.0,
            beta=0.0,
            num_sms=num_sms,
        ),
        dry_runs=warmup_reps,
        num_iters=reps,
        l2_flush=True,
        l2_flush_size_mb=256,
        l2_flush_device=torch.device("cuda:0"),
    )
    ms = np.median(measurements)

    # matmul: 2 * M * N * K
    # scale and add: 3 * M * N
    flops = (2 * M * N * K + 3 * M * N) / ms / 1e9
    print(
        f"GEMM Descriptor | num_sms: {num_sms}, M: {M}, N: {N}, K: {K}, {dtype}: {flops:.3f} TFLOPs/s"
    )


if __name__ == "__main__":
    assert supports_tma()

    for M, N, K in [(4096, 4096, 4096), (8192, 8192, 8192)]:
        for dtype in [
            torch.float8_e4m3fn,
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ]:
            for num_sms in [1, 16, 32, 64, 128, 132, 133, 256]:
                bench_gemm_persistent(num_sms, dtype, M, N, K)
                bench_gemm_descriptor_persistent(num_sms, dtype, M, N, K)
