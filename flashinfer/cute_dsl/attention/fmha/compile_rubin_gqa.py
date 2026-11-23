# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compile persistent Rubin GQA prefill kernels.

This compiler uses a fixed-length ABI separate from the ragged FMHA kernel.
CuTe and the vendored kernel are loaded only when compilation is requested.
"""

import functools
import math

import torch


def compile_rubin_gqa_kernel(
    num_qo_heads: int,
    num_kv_heads: int,
    is_causal: bool,
    with_lse: bool,
    device: torch.device,
):
    """Return a TVM-FFI handle and mask for padded E4M3 192/128 prefill.

    The specialization has fixed head counts, mask and LSE presence, with
    symbolic batch and sequence dimensions. Only SM107, persistent scheduling,
    FP32 softmax with unconditional rescaling, no split KV, and BF16 output
    are selected. The handle expects
    contiguous BSHD tensors. The CuTe DSL ragged batch wrapper prepares these
    handles during plan() for uniform grouped-head inputs, before graph capture.
    """
    if any(
        not isinstance(heads, int) or isinstance(heads, bool) or heads <= 0
        for heads in (num_qo_heads, num_kv_heads)
    ):
        raise ValueError("Rubin GQA requires positive integer head counts")
    if num_qo_heads % num_kv_heads or num_qo_heads // num_kv_heads not in (1, 2, 4, 8):
        raise ValueError("Rubin GQA requires a Q/KV head ratio of 1, 2, 4 or 8")
    if not isinstance(is_causal, bool) or not isinstance(with_lse, bool):
        raise ValueError("is_causal and with_lse must be bool")
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("Rubin GQA requires a CUDA SM107 device")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if torch.cuda.get_device_capability(device) != (10, 7):
        raise ValueError("Rubin GQA requires SM107 (compute capability 10.7)")
    return _compile_rubin_gqa_kernel_cached(
        num_qo_heads, num_kv_heads, is_causal, with_lse, device
    )


@functools.cache
def _compile_rubin_gqa_kernel_cached(
    num_qo_heads, num_kv_heads, is_causal, with_lse, device
):
    with torch.cuda.device(device):
        import cutlass
        import cutlass.cute as cute
        from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

        from .rubin_gqa.gqa import GqaTcgen05
        from .rubin_gqa.mask import CausalMask, DenseMask

        grouped_tile = num_qo_heads // num_kv_heads
        kernel = GqaTcgen05(
            (192, 128),
            grouped_tile,
            128 // grouped_tile,
            page_size=None,
            kv_splits=1,
            reduction_mode="none",
            softmax_dtype=cutlass.Float32,
            softmax_rescale_threshold=None,
            disable_persistence=False,
            disable_mma_rowsum=True,
        )
        mask = CausalMask() if is_causal else DenseMask()
        b, sq, sk = (cute.sym_int() for _ in range(3))

        def fake(dtype, shape):
            return make_fake_compact_tensor(
                dtype,
                shape,
                stride_order=tuple(reversed(range(len(shape)))),
                assumed_align=16,
            )

        q = fake(cutlass.Float8E4M3FN, (b, sq, num_qo_heads, 192))
        k = fake(cutlass.Float8E4M3FN, (b, sk, num_kv_heads, 192))
        v = fake(cutlass.Float8E4M3FN, (b, sk, num_kv_heads, 128))
        out = fake(cutlass.BFloat16, (b, sq, num_qo_heads, 128))
        lse = fake(cutlass.Float32, (b, sq, num_qo_heads)) if with_lse else None
        handle = cute.compile(
            kernel,
            None,  # Full KV lengths from the contiguous tensor shape.
            None,
            None,
            q,
            k,
            v,
            out,
            lse,
            None,
            None,
            None,
            None,
            mask,
            1.0 / math.sqrt(192),
            1.0,
            make_fake_stream(use_tvm_ffi_env_stream=True),
            False,
            None,
            options="--enable-tvm-ffi --opt-level 2",
        )
        return handle, mask
