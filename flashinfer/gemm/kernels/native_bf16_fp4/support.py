# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Metadata-only checks for the native-layout SM12x W4A16 backend."""

import torch

from ....utils import supported_compute_capability


def check_weights(b, b_descale, alpha, block_size=16):
    if block_size != 16:
        raise ValueError("cute-dsl-native requires block_size=16")
    if b.ndim != 2 or b.dtype != torch.uint8 or not b.is_contiguous():
        raise ValueError("cute-dsl-native requires contiguous uint8 weights (N,K/2)")
    n, k = b.shape[0], b.shape[1] * 2
    if min(n, k) <= 0 or k % 16:
        raise ValueError("cute-dsl-native requires positive N/K and K divisible by 16")
    if n * k >= 2**31:
        raise ValueError("cute-dsl-native requires N*K below 2**31")
    sf_bytes = ((n + 127) // 128) * ((k // 16 + 3) // 4) * 512
    if (
        b_descale.dtype not in (torch.uint8, torch.float8_e4m3fn)
        or not b_descale.is_contiguous()
        or b_descale.numel() != sf_bytes
    ):
        raise ValueError("cute-dsl-native requires the compact 128x4 E4M3 scale buffer")
    if b.data_ptr() % 4:
        raise ValueError("cute-dsl-native requires 4-byte aligned weights")
    tensors = [b, b_descale]
    if alpha is not None:
        if alpha.dtype != torch.float32 or alpha.shape != (1,):
            raise ValueError("cute-dsl-native alpha must be float32 with shape (1,)")
        tensors.append(alpha)
    if b.device.type != "cuda" or any(t.device != b.device for t in tensors):
        raise ValueError("cute-dsl-native requires operands on one CUDA device")
    return n, k


@supported_compute_capability([120, 121])
def check_native_bf16_fp4(
    a,
    b,
    b_descale,
    alpha=None,
    *,
    backend,
    out_dtype=None,
    out=None,
    block_size=16,
    enable_pdl=True,
):
    # Other backends consume differently prepared weights. Selection remains
    # explicit until the caller can describe which representation it owns.
    if backend == "auto":
        return False
    n, k = check_weights(b, b_descale, alpha, block_size)
    if (
        a.ndim != 2
        or a.dtype != torch.bfloat16
        or a.shape[1] != k
        or not 1 <= a.shape[0] <= 16
        or not a.is_contiguous()
        or a.device != b.device
    ):
        raise ValueError(
            "cute-dsl-native requires contiguous BF16 A[M,K], 1 <= M <= 16"
        )
    if a.numel() >= 2**31:
        raise ValueError("cute-dsl-native requires A below 2**31 elements")
    dtype = out_dtype or a.dtype
    if dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("cute-dsl-native requires BF16 or FP16 output")
    if out is not None:
        if (
            out.shape != (a.shape[0], n)
            or out.dtype != dtype
            or not out.is_contiguous()
            or out.device != a.device
        ):
            raise ValueError(
                "cute-dsl-native requires contiguous out[M,N] on A's device"
            )
        lo, hi = out.data_ptr(), out.data_ptr() + out.numel() * out.element_size()
        for t in (a, b, b_descale, alpha):
            if t is not None:
                start, end = t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()
                if lo < end and start < hi:
                    raise ValueError("cute-dsl-native output must not overlap an input")
    return True
