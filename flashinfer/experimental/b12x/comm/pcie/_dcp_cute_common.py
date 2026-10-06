"""CuTe DSL primitives shared by the PCIe DCP collectives.

The synchronization operations in this module intentionally spell the same
PTX instructions as the former CUDA implementations.  In particular, these
are system-scope operations: ordinary CuTe tensor loads/stores are not a safe
substitute for CUDA-IPC communication between devices.
"""

from __future__ import annotations

from typing import Sequence

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm


_FLAG_STRIDE = 32
_MAX_RANKS = 16


@dsl_user_op
def _membar_sys(*, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [],
        "membar.sys;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _store_relaxed_sys_u32(
    ptr: cute.Pointer,
    value: Uint32,
    *,
    loc=None,
    ip=None,
) -> None:
    llvm.inline_asm(
        None,
        [
            Uint32(value).ir_value(loc=loc, ip=ip),
            ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
        ],
        "st.relaxed.sys.global.u32 [$1], $0;",
        "r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _load_relaxed_sys_u32(
    ptr: cute.Pointer,
    *,
    loc=None,
    ip=None,
) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)],
            "ld.relaxed.sys.global.u32 $0, [$1];",
            "=r,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def block_pair_barrier(
    signals: Sequence[cute.Pointer],
    *,
    self_signal: cute.Pointer,
    rank: int,
    world_size: int,
    max_blocks: int,
) -> None:
    """Match the native per-block, per-peer double-buffered barrier."""
    cute.arch.sync_threads()
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    # Select the destination pointer once, then share one barrier body across
    # the participating lanes.  Keeping the body outside the constexpr loop
    # matches CUDA's runtime-indexed RankSignals path without cloning the
    # fence/store/poll sequence once per peer.
    if Int32(tidx) < Int32(world_size):
        peer_signal_address = Int64(signals[0].toint())
        for peer in cutlass.range_constexpr(1, world_size):
            if Int32(tidx) == Int32(peer):
                peer_signal_address = Int64(signals[peer].toint())
        peer_signal = cute.make_ptr(
            Uint32,
            peer_signal_address,
            cute.AddressSpace.gmem,
            assumed_align=4,
        )

        # CUDA's __threadfence_system precedes the local generation update.
        # All scaled address calculations remain Int64.
        _membar_sys()
        self_index = Int64(bidx) * Int64(_MAX_RANKS) + Int64(tidx)
        self_ptr = self_signal + self_index
        value = Uint32(cute.arch.load(self_ptr, Uint32)) + Uint32(1)
        cute.arch.store(self_ptr, value)

        slot = Int64(value % Uint32(2))
        common_index = (
            Int64(max_blocks * _MAX_RANKS)
            + (
                (slot * Int64(max_blocks) + Int64(bidx))
                * Int64(_MAX_RANKS * _FLAG_STRIDE)
            )
        )
        peer_index = common_index + Int64(rank * _FLAG_STRIDE)
        mine_index = common_index + Int64(tidx) * Int64(_FLAG_STRIDE)
        _store_relaxed_sys_u32(peer_signal + peer_index, value)
        observed = _load_relaxed_sys_u32(self_signal + mine_index)
        while observed != value:
            observed = _load_relaxed_sys_u32(self_signal + mine_index)

    cute.arch.sync_threads()


def signal_bytes(max_blocks: int) -> int:
    """Return the exact native ``Signal`` layout size."""
    return (
        max_blocks * _MAX_RANKS
        + 2 * max_blocks * _MAX_RANKS * _FLAG_STRIDE
    ) * 4


__all__ = ["block_pair_barrier", "signal_bytes"]
