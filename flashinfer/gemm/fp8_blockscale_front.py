# SPDX-FileCopyrightText: Copyright (c) 2026 FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Generalized front for fused-quantization FP8 block-scale GEMM.

``fp8_blockscale_gemm`` is the arch-neutral name for the op
:func:`flashinfer.gemm.fp8_blockscale_gemm_sm90` implements: a dense GEMM
whose activation side may arrive UNQUANTIZED (BF16) and is quantized to FP8
with 1x128 per-token-group scales inside the call, against an FP8 (or BF16)
weight with 1x128 or 128x128 block scales. That fused-quant contract is
what distinguishes this op from the pre-quantized FP8 facades
(:func:`gemm_fp8_nt_blockscaled` / :func:`gemm_fp8_nt_groupwise`), which
require ``a`` to already be an FP8 tensor with its scale grid computed by
the caller — a different contract, not a backend of this one.

The front owns no kernels and renames nothing: ``fp8_blockscale_gemm_sm90``
stays exactly as it is and is called underneath. Today SM90 is the only
architecture with an implementation of the fused-quant contract, so the
front's job is naming and routing headroom: new arch backends join as
plain strings without another arch-suffixed public entry.

Backends
--------
``"deepgemm"``
    The stock SM90 route of ``fp8_blockscale_gemm_sm90`` (its original
    DeepGEMM-based kernels), i.e. the underlying entry with
    ``backend="deepgemm"``.
``"specialized"``
    The agent-generated SM90 decode-M kernel, i.e. the underlying entry
    with ``backend="specialized"`` (that entry's own composable fallback
    and kill-switch semantics apply).
``"auto"``
    On SM90, the underlying entry's own ``backend="auto"`` (stock, with
    the specialized decode-M kernel on its packaged allowlist). On any
    other architecture it raises: no backend implements the fused-quant
    contract there. An explicitly selected backend is never silently
    swapped.
"""

from typing import Optional

import torch

from ..api_logging import flashinfer_api

# Backends behind the generalized front (plain strings, #5667 pattern).
FP8_BLOCKSCALE_GEMM_BACKENDS = ("auto", "deepgemm", "specialized")

# front backend -> fp8_blockscale_gemm_sm90 backend
_INNER_BACKEND = {"auto": "auto", "deepgemm": "deepgemm", "specialized": "specialized"}


@flashinfer_api
def fp8_blockscale_gemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    weight_scale: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    *,
    backend: str = "auto",
) -> torch.Tensor:
    """Fused-quantization FP8 block-scale GEMM (generalized front).

    Computes ``input @ weight.T`` with FP8 block-scaled arithmetic,
    quantizing a BF16 ``input`` internally (1x128 per-token groups). The
    parameters and dtype combinations are exactly those of
    :func:`fp8_blockscale_gemm_sm90`, which this front calls underneath —
    see that entry for the full contract (shapes, scale granularities,
    supported dtype combinations).

    Parameters
    ----------
    input : torch.Tensor
        Activation tensor of shape ``(M, K)``: BF16 (quantized internally)
        or FP8 with ``input_scale``.
    weight : torch.Tensor
        Weight tensor of shape ``(N, K)``: FP8 with ``weight_scale``, or
        BF16 (quantized internally).
    input_scale, weight_scale : torch.Tensor, optional
        Per-token-group / block scale grids, as in
        :func:`fp8_blockscale_gemm_sm90`.
    out, out_dtype
        As in :func:`fp8_blockscale_gemm_sm90`.
    backend : str
        One of ``FP8_BLOCKSCALE_GEMM_BACKENDS = ("auto", "deepgemm",
        "specialized")``. ``"deepgemm"`` pins the stock SM90 route,
        ``"specialized"`` the agent-generated SM90 decode-M kernel (with
        that backend's own composable fallback), and ``"auto"`` the
        underlying entry's own auto routing. Unknown names raise; an
        explicitly selected backend is never silently swapped.

    Returns
    -------
    torch.Tensor
        Output tensor of shape ``(M, N)``.

    Notes
    -----
    Pre-quantized FP8 inputs with caller-computed block scales on
    SM100+ are a different contract served by
    :func:`gemm_fp8_nt_blockscaled` / :func:`gemm_fp8_nt_groupwise`; this
    front intentionally does not route to them.
    """
    if backend not in FP8_BLOCKSCALE_GEMM_BACKENDS:
        raise ValueError(
            f"fp8_blockscale_gemm supports backend in "
            f"{FP8_BLOCKSCALE_GEMM_BACKENDS}, got {backend!r}; an "
            "explicitly selected backend is never silently swapped"
        )
    from .gemm_base import _match_sm_version, fp8_blockscale_gemm_sm90

    if not _match_sm_version(input.device, ["90", "90a"]):
        raise ValueError(
            "fp8_blockscale_gemm: the fused-quantization FP8 block-scale "
            "GEMM contract is currently implemented on SM90 only (backends "
            "'sm90' and 'specialized'). For pre-quantized FP8 inputs with "
            "caller-computed block scales on newer architectures, use "
            "gemm_fp8_nt_blockscaled / gemm_fp8_nt_groupwise (a different "
            "contract: no fused activation quantization)."
        )
    return fp8_blockscale_gemm_sm90(
        input,
        weight,
        input_scale,
        weight_scale,
        out,
        out_dtype,
        backend=_INNER_BACKEND[backend],
    )
