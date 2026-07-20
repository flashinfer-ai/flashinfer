# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
# Ported from b12x b12x/moe/fused/w4a8/weights.py @ 731ba9da (2026-06-30) -- one-time curated port.
# Upstream b12x is a research sandbox; this tree is the canonical home.
"""Prepared weight layout shared by the unified W4A8 MoE kernel."""

from __future__ import annotations

import torch


_TILE_N = 256
_TILE_K = 128


def repack_w4a8_weights(
    w_fp4: torch.Tensor,
    w_sf: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Repack logical MXFP4 weights for the dynamic W4A8 mainloop.

    ``w_fp4`` is ``[E, N, K/2]`` packed FP4 and ``w_sf`` is the matching
    ``[E, N, K/32]`` E8M0 grid.  The returned tensors use the N256/K128
    lane-major layouts consumed by :class:`MoEDynamicKernelBackend`.

    This is a representation transform, not a standalone GEMM contract.  The
    CUDA preparation path in :mod:`flashinfer.experimental.sm12x.moe.fused_moe._impl` performs the same
    permutation in-place so serving can reuse checkpoint storage.
    """

    if w_fp4.dim() != 3 or w_sf.dim() != 3:
        raise ValueError("w_fp4 must be [E, N, K/2] and w_sf [E, N, K/32]")
    e, n, k_half = w_fp4.shape
    k = k_half * 2
    if w_sf.shape != (e, n, k // 32):
        raise ValueError(f"w_sf shape {tuple(w_sf.shape)} != {(e, n, k // 32)}")
    if n % _TILE_N != 0:
        raise ValueError(f"N must be divisible by {_TILE_N}, got {n}")
    if k % _TILE_K != 0:
        raise ValueError(f"K must be divisible by {_TILE_K}, got {k}")
    n_tiles = n // _TILE_N
    k_tiles = k // _TILE_K

    b_u32 = w_fp4.contiguous().view(torch.int32).reshape(e, n, k // 8)
    b = b_u32.reshape(e, n_tiles, 8, 4, 8, k_tiles, 4, 4)
    b_rp = (
        b.permute(0, 1, 5, 6, 2, 4, 7, 3)
        .reshape(e, n_tiles, k_tiles, 4, 8, 32, 4)
        .contiguous()
    )

    sf = w_sf.contiguous().reshape(e, n_tiles, 32, 8, k_tiles, 4)
    sfb = (
        sf.permute(0, 1, 4, 2, 3, 5)
        .contiguous()
        .view(torch.int32)
        .reshape(e, n_tiles, k_tiles, 32, 8)
    )
    return b_rp, sfb


__all__ = ["repack_w4a8_weights"]
