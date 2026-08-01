# Copyright (c) 2025, Tri Dao.

from typing import Callable, TypeAlias

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Uint32, const_expr

from . import utils

MaskGenFn: TypeAlias = Callable[[int], Uint32]
MASK_R2P_CHUNK_SIZE: int = 32


@cute.jit
def r2p_bitmask_below(limit: Int32, s: int) -> Uint32:
    """32-bit R2P bitmask keeping positions < limit (exclusive upper bound)."""
    m = max((s + 1) * MASK_R2P_CHUNK_SIZE - limit, 0)
    return utils.shr_u32(Uint32(0xFFFFFFFF), Uint32(m))


@cute.jit
def mask_r2p_lambda(
    X: cute.Tensor,
    mask_gen_fn: cutlass.Constexpr[MaskGenFn],
    rank1: bool = False,
) -> None:
    """Apply R2P masking with a custom bitmask generator."""
    ncol = const_expr(
        cute.size(X.shape[cute.rank(X) - 1]) if not rank1 else cute.size(X.shape)
    )
    CHUNK_SIZE = MASK_R2P_CHUNK_SIZE
    for s in cutlass.range_constexpr(cute.ceil_div(ncol, CHUNK_SIZE)):
        mask = mask_gen_fn(s)
        for i in cutlass.range_constexpr(min(CHUNK_SIZE, ncol - s * CHUNK_SIZE)):
            in_bound = cutlass.Boolean(mask & (Uint32(1) << i))
            c = s * CHUNK_SIZE + i
            if const_expr(rank1):
                X[c] = X[c] if in_bound else -Float32.inf
            else:
                for r in cutlass.range_constexpr(cute.size(X.shape[0])):
                    X[r, c] = X[r, c] if in_bound else -Float32.inf


@cute.jit
def apply_block_size_mask(
    acc_S: cute.Tensor,
    block_size: Int32,
    n_block_size: cutlass.Constexpr[int] = 128,
) -> None:
    """Apply R2P bitmask masking positions >= block_size within a tile."""
    if block_size < n_block_size:
        mask_r2p_lambda(
            acc_S,
            lambda s: r2p_bitmask_below(block_size, s),
            rank1=True,
        )
