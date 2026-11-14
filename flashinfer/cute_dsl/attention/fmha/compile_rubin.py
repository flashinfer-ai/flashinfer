# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""JIT compiler for standard Rubin non-absorbed MLA prefill.

This compiler is separate from the Blackwell-derived FMHA compiler and artifact
loader. The CuTe DSL ragged prefill dispatcher selects it for supported SM107
inputs. Importing this module does not initialize CuTe or compile a kernel.
"""

import functools
import math

import torch


def compile_rubin_fmha_kernel(
    qk_dtype: torch.dtype,
    v_dtype: torch.dtype,
    out_dtype: torch.dtype,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    head_dim_v: int,
    is_causal: bool,
    with_lse: bool,
    enable_sink: bool,
    enable_skip_softmax: bool,
    use_pdl: bool,
    device: torch.device,
    has_window_left: bool = False,
    has_window_right=None,
):
    """Compile standard FP8 192/128 ragged attention on SM107.

    Supports E4M3 Q/K/V, BF16 output, equal positive Q/KV head counts, scalar
    scales, and optional FP32 base2 LSE. Causal masking is bottom-right aligned.
    Sinks, sliding windows, skip-softmax, PDL and optional Rubin numerical modes
    are not supported by this entry point. Unsupported configurations raise
    ``ValueError`` instead of selecting another implementation.

    The result is cached per concrete CUDA device, head count, mask and LSE
    presence. Batch size and sequence lengths remain symbolic. Compile before
    CUDA graph capture, and pass matching arguments to the existing ragged
    wrapper with ``enable_tvm_ffi=True``. The wrapper's compact NHD layout and
    output alignment requirements still apply. Empty-KV rows are skipped by
    the kernel; callers need the high-level wrapper's neutralization policy.
    """
    if (qk_dtype, v_dtype, out_dtype) != (
        torch.float8_e4m3fn,
        torch.float8_e4m3fn,
        torch.bfloat16,
    ):
        raise ValueError("Rubin FMHA requires E4M3 Q/K/V and BF16 output")
    if (head_dim, head_dim_v) != (192, 128):
        raise ValueError("Rubin FMHA requires head_dim=192 and head_dim_v=128")
    if (
        not isinstance(num_qo_heads, int)
        or isinstance(num_qo_heads, bool)
        or not isinstance(num_kv_heads, int)
        or isinstance(num_kv_heads, bool)
        or num_qo_heads <= 0
        or num_qo_heads != num_kv_heads
    ):
        raise ValueError("Rubin FMHA requires equal positive Q and KV head counts")
    if enable_sink or enable_skip_softmax or use_pdl:
        raise ValueError("Rubin FMHA does not support sinks, skip-softmax or PDL")
    if has_window_right is None:
        has_window_right = is_causal
    if has_window_left or bool(has_window_right) != bool(is_causal):
        raise ValueError(
            "Rubin FMHA only supports noncausal or bottom-right causal masks"
        )
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("Rubin FMHA requires a CUDA SM107 device")
    # Resolve the current device before looking up the cache: an unindexed
    # 'cuda' key would otherwise reuse a handle after a current-device change.
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if torch.cuda.get_device_capability(device) != (10, 7):
        raise ValueError("Rubin FMHA requires SM107 (compute capability 10.7)")
    return _compile_rubin_fmha_kernel_cached(
        num_qo_heads, bool(is_causal), bool(with_lse), device
    )


@functools.cache
def _compile_rubin_fmha_kernel_cached(num_heads, is_causal, with_lse, device):
    # CuTe initialization and the kernel constructor also consult device/arch.
    with torch.cuda.device(device):
        # Keep CuTe and the architecture-specific source out of ordinary imports.
        import cutlass
        import cutlass.cute as cute
        from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
        from cutlass.cute.typing import Int32

        from .rubin.helpers.fmha_helpers import MaskEnum
        from .rubin.kernel import RubinFusedMultiHeadAttentionForward

        kernel = RubinFusedMultiHeadAttentionForward(
            qk_acc_dtype=cutlass.Float32,
            pv_acc_dtype=cutlass.Float32,
            mma_tiler=(128, 128),
            head_dim=(192, 128),
            is_persistent=False,
            mask_type=(
                MaskEnum.WINDOW_MASK_INFERENCE if is_causal else MaskEnum.RESIDUAL_MASK
            ),
            enable_ex2_emulation=False,
            enable_skip_correction=True,
            rescale_threshold=8.0,
            use_tma_store=False,
        )
        b, sq, sk, hk, hr, seq = (cute.sym_int() for _ in range(6))

        def fake(dtype, shape, align=16):
            return make_fake_compact_tensor(
                dtype,
                shape,
                stride_order=tuple(reversed(range(len(shape)))),
                assumed_align=align,
            )

        q = fake(cutlass.Float8E4M3FN, (b, sq, hk, hr, 192))
        k = fake(cutlass.Float8E4M3FN, (b, sk, hk, 1, 192))
        v = fake(cutlass.Float8E4M3FN, (b, sk, hk, 1, 128))
        o = fake(cutlass.BFloat16, (b, sq, hk, hr, 128), 32)
        lse = fake(cutlass.Float32, (b, sq, hk, hr)) if with_lse else None
        cum_q, cum_k = fake(Int32, (seq,)), fake(Int32, (seq,))
        scale = 1 / math.sqrt(192)
        return cute.compile(
            kernel,
            q,
            k,
            v,
            o,
            (1, 1, 1, 1, num_heads, num_heads, 192, 128),
            cum_q,
            cum_k,
            lse,
            None,
            scale * math.log2(math.e),
            scale,
            1.0,
            None,
            None,
            Int32(0) if is_causal else None,
            None,
            None,
            make_fake_stream(use_tvm_ffi_env_stream=True),
            False,
            None,
            None,
            None,
            None,
            None,
            options="--enable-tvm-ffi --opt-level 2",
        )
