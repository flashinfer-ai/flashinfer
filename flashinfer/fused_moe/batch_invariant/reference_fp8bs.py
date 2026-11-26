# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""CPU references of the block-scale fp8 classes: the ``fp8bs.quant`` class vectorised on :class:`.reference.TorchF32`
(float32 CPU tensors, every op a single IEEE rounding, the e4m3 conversion the exact ``e4m3_round``) and exact on
:class:`.reference.ExactF32` for one block; the gate / up interleave's column map (an interleaved weight column reads its
original row's 128x128 scale block); the MoE class's GEMM1 chains on promoted accumulators (the bf16 classes' references).
The block amax is torch's ``amax`` (exact: the max of non-negative values has no rounding, so every order gives the tree's
bits). The promoted fold itself is the dense block-scale class's exact model (the device block partials promoted in exact
fp32 arithmetic: ``tests/gemm/test_batch_invariant_gemm_fp8_blockscale.py``).
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch

from ...batch_invariant.fp_interface import exact as fp32
from ...batch_invariant.quantization.nvfp4_reference import e4m3_bytes
from .blocks.epilogue import GROUP
from .reference import ExactF32, TorchF32
from .spec_fp8bs import (
    BLOCK,
    BLOCKS,
    block_chain,
    block_reduction,
    check_block,
    quant_elem,
)


def _f32(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to("cpu", torch.float32)


def quantize_rows_reference(
    x: torch.Tensor, scale_layout: str = "MN", block: int = BLOCK
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The class on rows ``x [M, K]`` (bf16 / f16 / f32; ``K % block == 0``): ``(codes, s)`` with ``codes`` the e4m3 values as an
    f32 ``[M, K]`` tensor (:func:`e4m3_bytes` gives the bytes) and ``s`` the fp32 scales, ``[K / block, M]`` (``"MN"``: the layout
    FlashInfer's DeepSeek activation pair carries and the block-scale GEMM's row-scale ring reads) or ``[M, K / block]`` (``"K"``);
    ``block`` 128, or 32 for the ``moe.e4m3bs32`` classes."""
    check_block(block)
    xf = _f32(x)
    M, K = xf.shape
    if K % block:
        raise ValueError(
            f"fp8bs.quant: K = {K} is not a whole number of {block}-element blocks"
        )
    blocks = xf.view(M, K // block, block)
    amax = TorchF32.fabs(blocks).amax(dim=-1, keepdim=True)  # exact
    s, inv = block_chain(TorchF32, amax)
    codes = quant_elem(TorchF32, blocks, inv).view(M, K)
    s = s.view(M, K // block)
    return codes, (s.t().contiguous() if scale_layout == "MN" else s.contiguous())


def quantize_rows_bytes(
    x: torch.Tensor, scale_layout: str = "MN", block: int = BLOCK
) -> Tuple[torch.Tensor, torch.Tensor]:
    """:func:`quantize_rows_reference` with the codes as uint8 bytes."""
    codes, s = quantize_rows_reference(x, scale_layout, block)
    return e4m3_bytes(codes), s


def quantize_weight_reference(
    w: torch.Tensor, block: int = BLOCK
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The class on a weight ``w [N, K]`` per ``block x block`` block: ``(codes [N, K] e4m3 values f32, s [N / block, K / block]
    fp32)``."""
    check_block(block)
    wf = _f32(w)
    N, K = wf.shape
    if N % block or K % block:
        raise ValueError(
            f"fp8bs.quant: a weight of shape ({N}, {K}) is not whole {block}x{block} blocks"
        )
    blocks = wf.view(N // block, block, K // block, block).permute(
        0, 2, 1, 3
    )  # (NB, KB, block, block)
    amax = TorchF32.fabs(blocks).amax(dim=(-1, -2), keepdim=True)
    s, inv = block_chain(TorchF32, amax)
    codes = quant_elem(TorchF32, blocks, inv).permute(0, 2, 1, 3).reshape(N, K)
    return codes, s[..., 0, 0].contiguous()


def exact_block(xs: Sequence[float]) -> Tuple[float, float, list]:
    """The scalar exact model of one block (128 or 32 Python floats, binary32 values): ``(s, inv, [e4m3 values])`` through the
    exact arithmetic, the canonical tree of the ``elem`` digit at the block's radix."""
    vals = [fp32.to_f32(float(v)) for v in xs]
    if len(vals) not in BLOCKS:
        raise ValueError(f"a block has one of {BLOCKS} elements, got {len(vals)}")
    amax = block_reduction(lambda fp, e: vals[e], len(vals)).canonical(ExactF32)
    s, inv = block_chain(ExactF32, amax)
    return s, inv, [quant_elem(ExactF32, v, inv) for v in vals]


def orig_rows(N1: int) -> torch.Tensor:
    """The original (canonical) weight row of every column of the gate / up interleaved GEMM1 weight (``N1 = 2I`` columns):
    column ``c`` holds up row ``8 (c // 16) + c % 8`` when ``c % 16 < 8``, gate row ``I + 8 (c // 16) + c % 16 - 8`` otherwise
    (:func:`.api.interleave_fc1`). Its 128-row scale block is ``orig_rows(N1) // 128``."""
    I = N1 // 2
    c = torch.arange(N1)
    q, r = c // (2 * GROUP), c % (2 * GROUP)
    return torch.where(r < GROUP, q * GROUP + r, I + q * GROUP + (r - GROUP))


def chain_cpu(acc: torch.Tensor, act, out_fmt: str = "bf16") -> torch.Tensor:
    """GEMM1's chain on promoted fp32 accumulators ``[rows, N1]`` (the interleaved (up, gate) pairs of the gated chains:
    ``[rows, I]`` out; every column of the ungated ones): the bf16 classes' vectorised references. ``act`` is an
    :class:`.spec.ActSpec`."""
    from . import reference as R

    a = _f32(acc)
    if not act.gated:
        return R.act_cpu(a, act.act, act.params, out_fmt)
    I = a.shape[1] // 2
    u = a.view(-1, I // GROUP, 2, GROUP)[:, :, 0, :].reshape(-1, I)
    g = a.view(-1, I // GROUP, 2, GROUP)[:, :, 1, :].reshape(-1, I)
    return R.glu_cpu_act(g, u, act.act, act.params, out_fmt)


__all__ = [
    "chain_cpu",
    "e4m3_bytes",
    "exact_block",
    "orig_rows",
    "quantize_rows_bytes",
    "quantize_rows_reference",
    "quantize_weight_reference",
]
