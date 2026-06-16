# SPDX-FileCopyrightText: Copyright (c) 2025 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""BF16 x FP4 (W4A16) dense GEMM public API.

Backend implementation details live in respective submodules
gemm_bf16_fp4_cudnn and gemm_bf16_fp4_cute_dsl.
"""

from typing import Literal, Optional, Tuple

import torch

from ..api_logging import flashinfer_api
from ..trace.templates.gemm import mm_bf16_fp4_trace_dispatch
from ..utils import backend_requirement, supported_compute_capability

from .gemm_base import (
    CUDNN_AVAILABLE,
    _check_cudnn_fp4_availability,
    _check_cute_dsl_availability,
)

if CUDNN_AVAILABLE:
    import cudnn


# Earliest cuDNN backend version supporting bf16 x fp4 GEMM
_CUDNN_BF16_FP4_MIN_BACKEND_VERSION = 92301


def _check_mm_bf16_fp4_problem_size(
    a: torch.Tensor,
    b: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    *,
    backend: Literal["cudnn", "cute-dsl"],
    out_dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
    block_size: int = 16,
    enable_pdl: bool = True,
):
    if a.dim() != 2:
        raise ValueError(f"a must be 2-D (M, K); got shape {tuple(a.shape)}")
    if a.dtype != torch.bfloat16:
        raise TypeError(
            f"a must be bfloat16; got {a.dtype}.  fp16 support is not implemented yet."
        )
    if out_dtype is not None and out_dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"out_dtype must be bfloat16 or float16; got {out_dtype}")
    if block_size != 16:
        raise ValueError(f"block_size must be 16 for FP4; got {block_size}")
    if alpha is not None and alpha.device != a.device:
        raise ValueError(
            f"alpha must be on the same device as a ({a.device}); got {alpha.device}"
        )
    return True


@supported_compute_capability([100, 103, 110, 120, 121])
def _cudnn_bf16_fp4_requirement(
    a: torch.Tensor,
    b: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    *,
    backend: Literal["cudnn", "cute-dsl"],
    out_dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
    block_size: int = 16,
    enable_pdl: bool = True,
):
    """cuDNN backend: requires a cuDNN build with FP4 block-scale support.

    Raises ``ValueError`` (not RuntimeError) so ``backend="auto"`` skips
    this backend instead of aborting (auto only catches ValueError).
    """
    if b.dtype != torch.uint8:
        raise ValueError(
            f"cudnn bf16 x fp4 expects the uint8 prepared weight from "
            f"prepare_bf16_fp4_weights(..., backend='cudnn'); got {b.dtype}."
        )
    _check_cudnn_fp4_availability()

    backend_version = cudnn.backend_version()
    if backend_version < _CUDNN_BF16_FP4_MIN_BACKEND_VERSION:
        raise ValueError(
            f"cuDNN bf16 x fp4 GEMM requires backend version >= "
            f"{_CUDNN_BF16_FP4_MIN_BACKEND_VERSION} (9.23.1), found {backend_version}. "
        )
    return True


@supported_compute_capability([100, 103, 110, 120, 121])
def _cute_dsl_bf16_fp4_requirement(
    a: torch.Tensor,
    b: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    *,
    backend: Literal["cudnn", "cute-dsl"],
    out_dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
    block_size: int = 16,
    enable_pdl: bool = True,
):
    if b.dtype != torch.int32:
        raise ValueError(
            f"cute-dsl bf16 x fp4 expects the int32 tile-packed weight from "
            f"prepare_bf16_fp4_weights(..., backend='cute-dsl'); got {b.dtype}."
        )
    _check_cute_dsl_availability()
    return True


# =============================================================================
# Public dispatchers
# =============================================================================


@flashinfer_api
def prepare_bf16_fp4_weights(
    b: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    *,
    backend: Literal["cudnn", "cute-dsl"],
    block_size: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Prepare FP4 weights for the bf16 x fp4 GEMM, for a specific backend.

    The caller is expected to start with weights in the canonical format
    that :func:`flashinfer.nvfp4_quantize` produces with
    ``sfLayout=layout_128x4``:

    * ``b`` is ``(N, K // 2)`` ``uint8`` with two FP4 codes packed per
      byte (low nibble = K=2i, high nibble = K=2i+1).
    * ``b_descale`` is the 128x4-swizzled FP8-E4M3 per-block scales,
      either as a 1-D byte buffer or a 2-D tensor.

    Each backend transforms these into whatever layout its compute kernel
    expects.  The returned ``(b, b_descale, alpha)`` tuple must be passed
    back to :func:`flashinfer.mm_bf16_fp4` with the *same* ``backend`` --
    the shapes / dtypes may not match other backends' expectations.

    Args:
        b: ``(N, K // 2)`` ``uint8`` packed FP4 weight.
        b_descale: 128x4-swizzled FP8-E4M3 scale factors from
            ``nvfp4_quantize``.  Either 1-D byte buffer or 2-D tensor.
        alpha: Optional ``(1,) float32`` global scalar.  Pass ``None``
            (default) for implicit ``alpha=1.0``.  Returned unchanged;
            forward the returned tuple to :func:`flashinfer.mm_bf16_fp4`.
        backend: Identifier of a supported backend (``"cudnn"`` or
            ``"cute-dsl"``).
        block_size: SF block size.  Always 16 for FP4.

    Returns:
        ``(b_prepared, b_descale_prepared, alpha_prepared)`` -- pass all
        three to :func:`flashinfer.mm_bf16_fp4` with the same ``backend``.

    Raises:
        ValueError: ``backend`` is unknown, or an input has an invalid
            shape (``b`` not 2-D, ``K`` not a multiple of ``block_size``,
            or ``alpha`` not shape ``(1,)``).
        TypeError: ``b`` is not ``uint8`` or ``alpha`` is not ``float32``.
    """
    if b.dim() != 2:
        raise ValueError(f"b must be 2-D (N, K/2); got shape {tuple(b.shape)}")
    if b.dtype != torch.uint8:
        raise TypeError(f"b must be uint8; got {b.dtype}")
    k = int(b.shape[1]) * 2
    if k % block_size != 0:
        raise ValueError(f"K={k} must be a multiple of block_size={block_size}")
    n = int(b.shape[0])
    k_sf = k // block_size
    expected_sf_bytes = ((n + 127) // 128) * ((k_sf + 3) // 4) * 512
    sf_bytes = b_descale.numel() * b_descale.element_size()
    if sf_bytes < expected_sf_bytes:
        raise ValueError(
            f"b_descale has {sf_bytes} bytes but the 128x4-swizzled layout for "
            f"N={n}, K_sf={k_sf} requires at least {expected_sf_bytes}"
        )
    if alpha is not None:
        if alpha.dim() != 1 or alpha.shape[0] != 1:
            raise ValueError(f"alpha must be shape (1,); got {tuple(alpha.shape)}")
        if alpha.dtype != torch.float32:
            raise TypeError(f"alpha must be float32; got {alpha.dtype}")
    if backend == "cudnn":
        from .gemm_bf16_fp4_cudnn import _prepare_cudnn

        return _prepare_cudnn(b, b_descale, alpha, block_size)
    if backend == "cute-dsl":
        from .gemm_bf16_fp4_cute_dsl import _prepare_cute_dsl

        return _prepare_cute_dsl(b, b_descale, alpha, block_size)
    raise ValueError(f"Unknown backend {backend!r}.  Supported: 'cudnn', 'cute-dsl'.")


@backend_requirement(
    {
        "cudnn": _cudnn_bf16_fp4_requirement,
        "cute-dsl": _cute_dsl_bf16_fp4_requirement,
    },
    common_check=_check_mm_bf16_fp4_problem_size,
)
@flashinfer_api(trace=mm_bf16_fp4_trace_dispatch)
def mm_bf16_fp4(
    a: torch.Tensor,
    b: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    *,
    backend: Literal["cudnn", "cute-dsl"],
    out_dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
    block_size: int = 16,
    enable_pdl: bool = True,
) -> torch.Tensor:
    """BF16 x FP4 GEMM: ``out = (a @ dequant(b).T) * alpha``.

    Intended to support **W4A16** workloads (4-bit weights, 16-bit activations)
    nvfp4 weights must be prepared for ``backend`` by
    :func:`prepare_bf16_fp4_weights`.  ``b``, ``b_descale``, and ``alpha``.

    Example:
        .. code-block:: python

            # 1) Prepare weights for a backend (once, at model load).
            b_p, sf_p, alpha_p = flashinfer.prepare_bf16_fp4_weights(
                b, b_descale, alpha, backend="cute-dsl",
            )
            # 2) Run the GEMM with the *same* backend tag.
            out = flashinfer.mm_bf16_fp4(
                a, b_p, sf_p, alpha_p, backend="cute-dsl",
            )

    Args:
        a: ``(M, K)`` activation matrix in ``torch.bfloat16``.  This is
            the only currently supported activation dtype; fp16 support
            can be added when needed.
        b: Prepared weight tensor (backend-specific layout).
        b_descale: Prepared scale-factor tensor (backend-specific layout).
        alpha: Optional ``(1,) float32`` global scalar.  Pass through
            whatever ``prepare_bf16_fp4_weights`` returned -- it may be
            ``None`` if the backend folded it into ``b_descale``.
        backend: Same identifier passed to ``prepare_bf16_fp4_weights``.
        out_dtype: Output dtype.  Defaults to ``a.dtype`` (``bfloat16``).
        out: Optional preallocated ``(M, N)`` output tensor.
        block_size: SF block size.  Always 16 for FP4.
        enable_pdl: Enable Programmatic Dependent Launch

    Returns:
        ``(M, N)`` tensor of ``out_dtype``.
    """
    out_dtype = out_dtype or a.dtype
    if backend == "cudnn":
        from .gemm_bf16_fp4_cudnn import _compute_cudnn

        return _compute_cudnn(a, b, b_descale, alpha, out_dtype, out, block_size)
    if backend == "cute-dsl":
        from .gemm_bf16_fp4_cute_dsl import _compute_cute_dsl

        return _compute_cute_dsl(
            a, b, b_descale, alpha, out_dtype, out, block_size, enable_pdl=enable_pdl
        )
    raise ValueError(f"Unknown backend {backend!r}.  Supported: 'cudnn', 'cute-dsl'.")


# =============================================================================
# Shared SF utility
# =============================================================================
#
# Used by both backends' prepare paths (cuDNN and cute-dsl) to turn the
# canonical 128x4-swizzled SF into a linear ``(N, K_sf)`` layout.


def _unswizzle_sf_128x4(sf_swizzled: torch.Tensor, n: int, k_sf: int) -> torch.Tensor:
    """Reverse the 128x4 SF swizzle into a flat ``(N, K_sf)`` byte tensor.

    The swizzle stores SF in 512-byte blocks each holding 128 N-rows x 4
    K_sf-cols.  The byte address of logical ``(n, k_sf)`` is::

        offset = ((n // 128) * sf_pad_blocks + k_sf // 4) * 512
               + (n % 32) * 16
               + ((n % 128) // 32) * 4
               + (k_sf % 4)

    where ``sf_pad_blocks = ceil(k_sf, 4) // 4`` accounts for K_sf
    padding inside each 128-row N block.
    """
    device = sf_swizzled.device
    sf_flat = sf_swizzled.contiguous().view(torch.uint8).view(-1)
    sf_pad_blocks = (k_sf + 3) // 4  # ceil_div(k_sf, 4)
    n_idx = torch.arange(n, device=device, dtype=torch.int64)
    k_idx = torch.arange(k_sf, device=device, dtype=torch.int64)
    n_grid, k_grid = torch.meshgrid(n_idx, k_idx, indexing="ij")
    offsets = (
        ((n_grid // 128) * sf_pad_blocks + (k_grid // 4)) * 512
        + (n_grid % 32) * 16
        + ((n_grid % 128) // 32) * 4
        + (k_grid % 4)
    )
    return sf_flat[offsets]


__all__ = [
    "prepare_bf16_fp4_weights",
    "mm_bf16_fp4",
]
