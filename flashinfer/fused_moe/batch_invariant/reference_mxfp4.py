# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Vectorised torch CPU models of the MXFP4 MoE classes' chains (:mod:`.spec_mxfp4`), term for term through ``TorchF32``:
GEMM1's bf16 chain on the raw fp32 accumulators (the bf16 class's chains of :mod:`.reference`), the intermediate's block
quantization by the ``mxfp4.quant`` class's reference (:mod:`flashinfer.batch_invariant.quantization.mx_reference`), GEMM2's
conversion. Bit-exact references for the device suite."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch

from ...batch_invariant.quantization.mx_reference import quantize_mx_reference
from ...batch_invariant.quantization.mx_spec import DEFAULT_RULE, MXFP4
from . import spec as S
from .reference import TorchF32, _f32


def mx_gemm1_cpu(
    g: torch.Tensor,
    u: torch.Tensor,
    act: str = "silu",
    params: Sequence[Optional[float]] = (),
) -> torch.Tensor:
    """The gated GEMM1 chain for whole rows: ``cvt_bf16(glu_pre(g, u))`` on the ``[rows, I]`` fp32 accumulators (the
    exponential and the divisions on the part's SFU, from its tables)."""
    return S.glu_elem(
        TorchF32, _f32(g), _f32(u), act, "bf16", params, S.hw_arith_applies(act, True)
    )


def mx_act_cpu(
    acc: torch.Tensor, act: str = "relu2", params: Sequence[Optional[float]] = ()
) -> torch.Tensor:
    """The ungated GEMM1 chain for whole rows: ``cvt_bf16(act(acc))``."""
    return S.act_elem(
        TorchF32, _f32(acc), act, "bf16", params, S.hw_arith_applies(act, False)
    )


def mx_gemm2_cpu(acc: torch.Tensor) -> torch.Tensor:
    """GEMM2's chain: ``cvt_bf16(acc)`` on the ``[rows, H]`` fp32 accumulators."""
    return TorchF32.cvt_out(_f32(acc), "bf16")


def mx_quantize_cpu(
    x: torch.Tensor, swizzled: str = "128x4"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The ``mxfp4.quant`` class's reference on bf16 rows ``[M, K]``: ``(packed codes [M, K/2], sf flat uint8 in the 128x4
    layout, sf [M, K/32] logical bytes)``."""
    return quantize_mx_reference(x, MXFP4, DEFAULT_RULE, swizzled)
