# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Lightweight admission checks; no kernel/compiler imports."""

import torch
from flashinfer.utils import experimental_backend, supported_compute_capability


@experimental_backend
@supported_compute_capability([100, 107])
def check_bmm_fp8(A, B, A_scale, B_scale, dtype, out=None, backend="auto"):
    if A.ndim != 3 or B.ndim != 3:
        raise ValueError("frost-low-latency requires rank-3 A and B")
    batch, m, k = A.shape
    if batch != 1 or B.shape[0] != 1 or B.shape[1] != k:
        raise ValueError("frost-low-latency requires batch dimension 1 and matching K")
    n = B.shape[2]
    if not (1 <= m <= 64 and k > 0 and k % 512 == 0 and n > 0 and n % 8 == 0):
        raise ValueError(
            "frost-low-latency requires 1 <= M <= 64, K % 512 == 0, N % 8 == 0"
        )
    if (
        A.dtype != torch.float8_e4m3fn
        or B.dtype != torch.float8_e4m3fn
        or dtype != torch.bfloat16
    ):
        raise ValueError("frost-low-latency requires E4M3 inputs and BF16 output")
    if not A.is_contiguous() or not B.transpose(-2, -1).is_contiguous():
        raise ValueError("frost-low-latency requires row-major A and column-major B")
    if not A.is_cuda or any(t.device != A.device for t in (B, A_scale, B_scale)):
        raise ValueError(
            "frost-low-latency requires all inputs on the same CUDA device"
        )
    if any(t.dtype != torch.float32 or t.numel() != 1 for t in (A_scale, B_scale)):
        raise ValueError("frost-low-latency requires scalar FP32 scales")
    if A.data_ptr() % 16 or B.data_ptr() % 16:
        raise ValueError("frost-low-latency requires 16-byte aligned A and B")
    if out is not None and (
        out.shape != (1, m, n)
        or out.dtype != dtype
        or out.device != A.device
        or not out.is_contiguous()
    ):
        raise ValueError(
            "frost-low-latency requires contiguous BF16 out with shape (1, M, N) on the input device"
        )
    return True
