# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""MXFP8 x MXFP4 (W4A8 in the MX formats) dense GEMM public API: e4m3 activations against packed e2m1 weights, one UE8M0
block scale per 32 elements on each operand, applied inside the mixed-operand block-scaled instruction.

The batch-invariant class of ``flashinfer/gemm/batch_invariant`` is the pair's only dense kernel in FlashInfer, so both
values of ``batch_invariant`` run it; see ``flashinfer/batch_invariant/README.md``.
"""

from typing import Optional

import torch

from ..api_logging import flashinfer_api


@flashinfer_api
def mm_mxfp8_mxfp4(
    a: torch.Tensor,
    b: torch.Tensor,
    a_descale: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    out: Optional[torch.Tensor] = None,
    *,
    batch_invariant: bool = False,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    r"""MM MXFP8 x MXFP4 (block size 32): ``out = cvt((a @ b) * alpha)`` for MXFP8
    activations against MXFP4 weights.

    Parameters
    ----------
    a: torch.Tensor
        Input A tensor, shape (m, k), mxfp8 e4m3 (``float8_e4m3fn``), row-major:
        the output of ``mxfp8_quantize(..., is_sf_swizzled_layout=True)``.

    b: torch.Tensor
        Input B tensor, shape (k // 2, n), packed mxfp4 e2m1 (uint8 or
        ``float4_e2m1fn_x2``, two elements per byte), should be column major: the
        transpose of the row-major (n, k // 2) weight ``mxfp4_quantize`` returns.
        The block-scaled instruction takes K-major operands only, so a row-major
        (k // 2, n) weight raises.

    a_descale: torch.Tensor
        Block scale tensor for A, uint8 (fp8 e8m0), 1D swizzled 128x4 layout:
        shape (M_padded * K_padded,) where M_padded = round_up(m, 128) and
        K_padded = round_up(k // 32, 4). Produced by
        ``mxfp8_quantize(..., is_sf_swizzled_layout=True)``.

    b_descale: torch.Tensor
        Block scale tensor for B, uint8 (fp8 e8m0), 1D swizzled 128x4 layout:
        shape (N_padded * K_padded,) where N_padded = round_up(n, 128) and
        K_padded = round_up(k // 32, 4). Produced by
        ``mxfp4_quantize(..., sfLayout=SfLayout.layout_128x4)``.

    alpha: Optional[torch.Tensor]
        A 1-element fp32 tensor on A's device: the product is multiplied by it
        once, after the whole K reduction (``None``: 1.0). A per-token alpha is
        not supported.

    out_dtype: torch.dtype
        Output dtype, bf16 or fp16. Defaults to ``torch.bfloat16``.

    out: Optional[torch.Tensor]
        Out tensor, shape (m, n), row-major, of ``out_dtype``. If provided, the
        result is written into it. Defaults to ``None``.

    batch_invariant: bool
        Both values run the batch-invariant mixed MX GEMM class today: FlashInfer
        has no other dense kernel for this operand pair, so ``False`` (the
        default) selects the same kernels as ``True``. ``True`` is the guarantee
        and keeps it: a row of the output is bit-identical whatever the number of
        rows, the row's position in the batch or the configuration chosen, and
        the result is deterministic; a future non-invariant kernel may take the
        ``False`` path. ``b`` must be the transpose of a row-major ``(n, k // 2)``
        weight. The class is the architecture's (the SM107 instruction folds 64
        elements per step, the SM100 / SM103 one 32), so the specification is
        ``flashinfer.batch_invariant.fingerprint("mm_mxfp8_mxfp4", N=, K=,
        out_dtype=, arch=)``. Unsupported inputs raise
        :class:`flashinfer.batch_invariant.BatchInvariantUnsupportedError`
        instead of falling back. See ``flashinfer/batch_invariant/README.md``.
        Defaults to ``False``.

    enable_pdl: Optional[bool]
        Programmatic dependent launch: ``None`` (the default) follows the device,
        ``True`` / ``False`` forces it. A placement: the same bits either way.

    Returns
    -------
    out: torch.Tensor
        Out tensor, shape (m, n), bf16 or fp16.

    Notes
    -----
    ``k`` is a multiple of 32 and ``n`` of 8 on every part; on SM100 / SM103 ``k``
    is a multiple of 128 (the instruction reads the e2m1 weight through the TMA's
    unpacking format in 128-element runs of the row; a smaller ``k`` raises, it is
    not padded). Supported on SM100 / SM103 / SM107.

    Examples
    --------
    >>> import torch
    >>> from flashinfer import SfLayout, mm_mxfp8_mxfp4, mxfp4_quantize, mxfp8_quantize
    >>> m, n, k = 512, 256, 1024
    >>> # Create input tensors - note: weight is [n, k] for typical NN layers
    >>> a = torch.randn([m, k], device="cuda", dtype=torch.bfloat16)
    >>> weight = torch.randn([n, k], device="cuda", dtype=torch.bfloat16)
    >>>
    >>> # Quantize input [m, k] to MXFP8 - scales are 1D swizzled for (M, K/32) layout
    >>> a_mx, a_sf = mxfp8_quantize(a, is_sf_swizzled_layout=True)
    >>> # Quantize weight [n, k] to MXFP4 (packed [n, k/2]) - scales 1D swizzled for (N, K/32)
    >>> w_mx, w_sf = mxfp4_quantize(weight, sfLayout=SfLayout.layout_128x4)
    >>> # Pass weight.T as [k/2, n] and the 1D swizzled scales directly
    >>> out = mm_mxfp8_mxfp4(a_mx, w_mx.t(), a_sf, w_sf, out_dtype=torch.bfloat16)
    >>> out.shape
    torch.Size([512, 256])
    """
    from ..batch_invariant.support import unsupported
    from .batch_invariant.api import (
        CLASS_MXW48,
        MM_MXW48_API_OUT_DTYPES,
        mm_mxfp8_mxfp4_batch_invariant,
    )

    if out_dtype not in MM_MXW48_API_OUT_DTYPES:
        raise unsupported(CLASS_MXW48, f"out_dtype={out_dtype}", "bf16 or fp16 output")
    # both flag values: the batch-invariant class is the pair's only dense kernel
    return mm_mxfp8_mxfp4_batch_invariant(
        a, b, a_descale, b_descale, alpha, out_dtype, out, enable_pdl=enable_pdl
    )


__all__ = [
    "mm_mxfp8_mxfp4",
]
