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

import numpy as np
import torch
import triton
import triton.language as tl

from flashinfer.gemm import gemm_fp8_nt_groupwise
from flashinfer.testing.utils import bench_gpu_time


@triton.jit
def _w8a8_block_fp8_matmul(
    # Pointers to inputs and output
    A,
    B,
    C,
    As,
    Bs,
    # Shape for matmul
    M,
    N,
    K,
    # Block size for block-wise quantization
    group_n,
    group_k,
    # Stride for inputs and output
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_As_m,
    stride_As_k,
    stride_Bs_k,
    stride_Bs_n,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Triton-accelerated function used to perform linear operations (dot
    product) on input tensors `A` and `B` with block-wise quantization, and store the result in output
    tensor `C`.
    """

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    As_ptrs = As + offs_am * stride_As_m
    offs_bsn = offs_bn // group_n
    Bs_ptrs = Bs + offs_bsn * stride_Bs_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        k_start = k * BLOCK_SIZE_K
        offs_ks = k_start // group_k
        a_s = tl.load(As_ptrs + offs_ks * stride_As_k)
        b_s = tl.load(Bs_ptrs + offs_ks * stride_Bs_k)

        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if C.dtype.element_ty == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        c = accumulator.to(tl.float16)
    else:
        c = accumulator.to(tl.float32)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def triton_w8a8_block_fp8_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    M = A.shape[0]
    N, K = B.shape
    block_n, block_k = 128, 128

    config = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": block_n,
        "BLOCK_SIZE_K": block_k,
        "GROUP_SIZE_M": 32,
        "num_warps": 4,
        "num_stages": 3,
    }

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    _w8a8_block_fp8_matmul[grid](
        A,
        B,
        out,
        As,
        Bs,
        M,
        N,
        K,
        block_n,
        block_k,
        A.stride(-2),
        A.stride(-1),
        B.stride(1),
        B.stride(0),
        out.stride(-2),
        out.stride(-1),
        As.stride(-2),
        As.stride(-1),
        Bs.stride(1),
        Bs.stride(0),
        **config,
    )

    return out


def bench_groupwise_gemm_fp8_blackwell(m, n, k, in_dtype, out_dtype):
    a = torch.randn((m, k), device="cuda").to(in_dtype)
    b = torch.randn((n, k), device="cuda").to(in_dtype)
    a_scale = torch.rand((k // 128, m), dtype=torch.float32, device="cuda")
    b_scale = torch.rand((k // 128, n // 128), dtype=torch.float32, device="cuda")

    out = torch.empty((m, n), dtype=out_dtype, device="cuda")
    gemm_fp8_nt_groupwise(a, b, a_scale, b_scale, out=out)

    measurements = bench_gpu_time(
        lambda: gemm_fp8_nt_groupwise(a, b, a_scale, b_scale, out=out),
        dry_runs=10,
        num_iters=100,
        l2_flush=True,
        l2_flush_size_mb=256,
        l2_flush_device=torch.device("cuda:0"),
    )
    ms = np.median(measurements)
    tflops_per_second = 2 * m * n * k * 1e-9 / ms
    print(
        f"gemm_fp8_nt_groupwise {m} {n} {k} {in_dtype} {out_dtype}: {tflops_per_second:.2f} TFLOPs/s"
    )

    tl_out = torch.empty((m, n), dtype=out_dtype, device="cuda")
    a_scale = a_scale.transpose(0, 1).contiguous()
    b_scale = b_scale.transpose(0, 1).contiguous()
    measurements = bench_gpu_time(
        lambda: triton_w8a8_block_fp8_matmul(a, b, a_scale, b_scale, tl_out),
        dry_runs=10,
        num_iters=100,
        l2_flush=True,
        l2_flush_size_mb=256,
        l2_flush_device=torch.device("cuda:0"),
    )
    ms = np.median(measurements)
    tflops_per_second = 2 * m * n * k * 1e-9 / ms
    print(
        f"triton_gemm_fp8_nt_groupwise {m} {n} {k} {in_dtype} {out_dtype}: {tflops_per_second:.2f} TFLOPs/s"
    )


if __name__ == "__main__":
    for m in [1024, 2048, 4096, 8192]:
        for n in [1024, 2048, 4096, 8192]:
            for k in [1024, 2048, 4096, 8192]:
                bench_groupwise_gemm_fp8_blackwell(
                    m, n, k, torch.float8_e5m2, torch.bfloat16
                )
