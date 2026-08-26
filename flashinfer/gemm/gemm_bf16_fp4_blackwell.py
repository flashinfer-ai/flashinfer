# SPDX-FileCopyrightText: Copyright (c) 2025 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Source-built Blackwell backends for the BF16 x FP4 GEMM."""

from typing import Literal, Optional, Tuple

import torch

from ..utils import get_compute_capability
from .gemm_bf16_fp4 import _unswizzle_sf_128x4
from .gemm_bf16_fp4_cute_dsl import (
    _cute_dsl_pack_fp4_weight,
    _e4m3_to_s0e5m3,
)

BlackwellBf16Fp4Backend = Literal["blackwell-native", "blackwell-tiled"]

_BLACKWELL_NATIVE_LAYOUT = 0
_BLACKWELL_TILED_LAYOUT = 1


def _require_blackwell_source_arch(device: torch.device) -> None:
    major, minor = get_compute_capability(device)
    if (major, minor) not in ((10, 0), (10, 3)):
        raise NotImplementedError(
            "the source-built BF16 x FP4 backends require SM100 or SM103; "
            f"got SM{major}{minor}"
        )


def _get_blackwell_bf16_fp4_module():
    """Load the source-built module lazily through the public JIT boundary."""
    from ..jit.gemm.blackwell_bf16_fp4 import get_blackwell_bf16_fp4_module

    return get_blackwell_bf16_fp4_module()


def _prepare_blackwell_bf16_fp4_alpha(
    alpha: Optional[torch.Tensor], a: torch.Tensor
) -> torch.Tensor:
    """Encode implicit alpha with an A alias; validate explicit alpha."""
    if alpha is None:
        return a.view(torch.float32).reshape(-1)[:1]
    if alpha.device != a.device:
        raise ValueError(
            f"alpha must be on the same device as the GEMM inputs ({a.device}); "
            f"got {alpha.device}"
        )
    if (
        alpha.dtype != torch.float32
        or tuple(alpha.shape) != (1,)
        or not alpha.is_contiguous()
    ):
        raise ValueError(
            "alpha must be a contiguous float32 tensor with shape (1,); "
            f"got dtype={alpha.dtype}, shape={tuple(alpha.shape)}, "
            f"contiguous={alpha.is_contiguous()}"
        )
    if alpha.data_ptr() == a.data_ptr():
        raise ValueError("explicit alpha must not alias a")
    return alpha


def _prepare_blackwell_bf16_fp4(
    b: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor],
    block_size: int,
    backend: BlackwellBf16Fp4Backend,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Prepare one of the two explicit source-kernel tensor layouts."""
    _require_blackwell_source_arch(b.device)
    if b.device != b_descale.device:
        raise ValueError(
            "b and b_descale must be on the same device; got "
            f"{b.device} and {b_descale.device}"
        )
    if alpha is not None and alpha.device != b.device:
        raise ValueError(
            f"alpha must be on the same device as b ({b.device}); got {alpha.device}"
        )

    n = int(b.shape[0])
    k = int(b.shape[1]) * 2
    k_sf = k // block_size
    linear_sf = _unswizzle_sf_128x4(b_descale, n, k_sf).contiguous()

    if backend == "blackwell-native":
        return b.contiguous(), linear_sf.view(torch.float8_e4m3fn), alpha

    if backend == "blackwell-tiled":
        if n % 64 != 0:
            raise ValueError(
                f"blackwell-tiled requires N to be a multiple of 64; got N={n}"
            )
        b_kn = b.t().contiguous()
        b_packed = _cute_dsl_pack_fp4_weight(b_kn)
        scale_s0e5m3 = _e4m3_to_s0e5m3(linear_sf.t().contiguous())
        return b_packed, scale_s0e5m3, alpha

    raise ValueError(f"unknown source-built BF16 x FP4 backend {backend!r}")


def _validate_blackwell_native_layout(
    a: torch.Tensor,
    b: torch.Tensor,
    b_descale: torch.Tensor,
) -> Tuple[int, int]:
    if b.dim() != 2 or b.dtype != torch.uint8:
        raise ValueError(
            "blackwell-native expects b as contiguous uint8 [N, K/2]; "
            f"got dtype={b.dtype}, shape={tuple(b.shape)}"
        )
    n = int(b.shape[0])
    k = int(b.shape[1]) * 2
    expected_scale_shape = (n, k // 16)
    if (
        b_descale.dtype != torch.float8_e4m3fn
        or tuple(b_descale.shape) != expected_scale_shape
    ):
        raise ValueError(
            "blackwell-native expects linear float8_e4m3fn scales "
            f"with shape {expected_scale_shape}; got dtype={b_descale.dtype}, "
            f"shape={tuple(b_descale.shape)}"
        )
    if int(a.shape[1]) != k:
        raise ValueError(
            f"a.shape[1]={int(a.shape[1])} but b.shape={tuple(b.shape)} encodes K={k}"
        )
    return n, _BLACKWELL_NATIVE_LAYOUT


def _validate_blackwell_tiled_layout(
    a: torch.Tensor,
    b: torch.Tensor,
    b_descale: torch.Tensor,
) -> Tuple[int, int]:
    if b.dim() != 2 or b.dtype != torch.int32 or int(b.shape[1]) % 2 != 0:
        raise ValueError(
            "blackwell-tiled expects b as contiguous int32 [K/16, N*2]; "
            f"got dtype={b.dtype}, shape={tuple(b.shape)}"
        )
    k_tiles = int(b.shape[0])
    n = int(b.shape[1]) // 2
    k = k_tiles * 16
    if n % 64 != 0:
        raise ValueError(
            f"blackwell-tiled requires N to be a multiple of 64; got N={n}"
        )
    expected_scale_shape = (k_tiles, n)
    if (
        b_descale.dtype != torch.uint8
        or tuple(b_descale.shape) != expected_scale_shape
    ):
        raise ValueError(
            "blackwell-tiled expects S0E5M3 uint8 scales "
            f"with shape {expected_scale_shape}; got dtype={b_descale.dtype}, "
            f"shape={tuple(b_descale.shape)}"
        )
    if int(a.shape[1]) != k:
        raise ValueError(
            f"a.shape[1]={int(a.shape[1])} but b.shape={tuple(b.shape)} encodes K={k}"
        )
    return n, _BLACKWELL_TILED_LAYOUT


def _compute_blackwell_bf16_fp4(
    a: torch.Tensor,
    b: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out: Optional[torch.Tensor],
    block_size: int,
    enable_pdl: bool,
    backend: BlackwellBf16Fp4Backend,
) -> torch.Tensor:
    """Validate the prepared ABI and launch the source-built dispatcher."""
    _require_blackwell_source_arch(a.device)
    if a.dim() != 2 or a.dtype != torch.bfloat16:
        raise ValueError(
            "source-built BF16 x FP4 GEMM expects a as bfloat16 [M, K]; "
            f"got dtype={a.dtype}, shape={tuple(a.shape)}"
        )
    if block_size != 16 or int(a.shape[1]) % block_size != 0:
        raise ValueError(
            f"source-built BF16 x FP4 GEMM requires block_size=16 and K%16=0; "
            f"got block_size={block_size}, K={int(a.shape[1])}"
        )
    if not a.is_contiguous() or not b.is_contiguous() or not b_descale.is_contiguous():
        raise ValueError("a, b, and b_descale must be contiguous")
    if b.device != a.device or b_descale.device != a.device:
        raise ValueError(
            "a, b, and b_descale must be on the same device; got "
            f"{a.device}, {b.device}, and {b_descale.device}"
        )

    if backend == "blackwell-native":
        n, layout_code = _validate_blackwell_native_layout(a, b, b_descale)
        if out_dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(
                "blackwell-native supports bfloat16 or float16 output; "
                f"got {out_dtype}"
            )
    elif backend == "blackwell-tiled":
        n, layout_code = _validate_blackwell_tiled_layout(a, b, b_descale)
        if out_dtype != torch.bfloat16:
            raise ValueError(
                f"blackwell-tiled requires bfloat16 output; got {out_dtype}"
            )
    else:
        raise ValueError(f"unknown source-built BF16 x FP4 backend {backend!r}")

    m = int(a.shape[0])
    if out is None:
        out = torch.empty((m, n), device=a.device, dtype=out_dtype)
    elif tuple(out.shape) != (m, n):
        raise ValueError(f"out shape {tuple(out.shape)} != expected {(m, n)}")
    elif out.dtype != out_dtype:
        raise TypeError(f"out dtype {out.dtype} != requested out_dtype {out_dtype}")
    elif out.device != a.device or not out.is_contiguous():
        raise ValueError(
            "out must be contiguous and on the same device as a; got "
            f"device={out.device}, contiguous={out.is_contiguous()}"
        )

    alpha_for_launch = _prepare_blackwell_bf16_fp4_alpha(alpha, a)
    b_descale_for_launch = (
        b_descale.view(torch.uint8)
        if layout_code == _BLACKWELL_NATIVE_LAYOUT
        else b_descale
    )
    module = _get_blackwell_bf16_fp4_module()
    module.run(
        a,
        b,
        b_descale_for_launch,
        alpha_for_launch,
        out,
        layout_code,
        bool(enable_pdl),
    )
    return out


__all__ = []
