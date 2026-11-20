"""Device address mapping for the by-value symmetric buffer parameter."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Uint32, Uint64

from ..ptx.ld_st import ld_peer_param_u64


@cute.jit
def mapped_peer_base(peer_info: cute.Tensor, rank: Uint32,
                     num_ranks: cutlass.Constexpr) -> Uint64:
    """Reference SymBuffer::map(base, rank) using signed relative bytes."""
    base = ld_peer_param_u64(peer_info.iterator)
    if cutlass.const_expr(num_ranks == 1):
        return base
    delta = ld_peer_param_u64(peer_info.iterator + Uint32(1) + rank)
    return base + delta
