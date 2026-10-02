"""Warp-level collectives: reductions, prefix scans, and ballot ranking.

Everything here operates within one warp of 32 lanes with all lanes participating; there is
no memory traffic and no block barrier.  These are the cheapest cross-thread operations on
the GPU (one `redux.sync` or five `shfl.sync` steps), so block-level algorithms are built to
do as much as possible at warp scope and cross warps only once through shared memory.

Type discipline: the reductions lower to `redux.sync.<op>.<s32|u32>` by the operand's DSL
type.  Pass a genuine ``cutlass.Uint32`` for unsigned min/max (ordered keys); an Int32 would
silently reduce as signed and mis-order keys with the top bit set.
"""

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

__all__ = [
    "warp_sum",
    "warp_max_u32",
    "warp_min_u32",
    "warp_inclusive_scan_add",
    "warp_exclusive_scan_add",
    "warp_suffix_scan_add",
    "warp_broadcast",
    "ballot_rank",
    "ballot_count",
    "match_any",
]


def warp_sum(v):
    """Sum over the 32 lanes, returned to every lane.  One ``redux.sync.add`` instruction.
    Int32 or Uint32 (bit-identical for u32)."""
    return cute.arch.warp_redux_sync(v, "add")


def warp_max_u32(v):
    """Unsigned maximum over the warp; ``v`` must be ``cutlass.Uint32``.  One instruction."""
    return cute.arch.warp_redux_sync(v, "max")


def warp_min_u32(v):
    """Unsigned minimum over the warp; ``v`` must be ``cutlass.Uint32``.  One instruction."""
    return cute.arch.warp_redux_sync(v, "min")


@cute.jit
def warp_inclusive_scan_add(val, lane):
    """Inclusive prefix sum: lane i receives val[0] + ... + val[i].

    Five ``shfl.up`` steps with the add gated on ``lane >= offset`` (the shuffle returns the
    lane's own value below the offset, which must not be added).  ~5 x shuffle latency.
    """
    for offset in [1, 2, 4, 8, 16]:
        other = cute.arch.shuffle_sync_up(val, offset, mask_and_clamp=0)
        if lane >= cutlass.Int32(offset):
            val = val + other
    return val


@cute.jit
def warp_exclusive_scan_add(val, lane):
    """Exclusive prefix sum and the warp total: lane i receives (val[0] + ... + val[i-1], sum).

    The inclusive scan minus the lane's own value; the total is lane 31's inclusive value,
    broadcast with one more shuffle.
    """
    inclusive = warp_inclusive_scan_add(val, lane)
    total = cute.arch.shuffle_sync(inclusive, cutlass.Int32(31))
    return inclusive - val, total


@cute.jit
def warp_suffix_scan_add(val, lane):
    """Inclusive suffix sum: lane i receives val[i] + ... + val[31].  Five ``shfl.down`` steps."""
    for offset in [1, 2, 4, 8, 16]:
        other = cute.arch.shuffle_sync_down(val, offset, mask_and_clamp=31)
        if lane + cutlass.Int32(offset) < cutlass.Int32(32):
            val = val + other
    return val


def warp_broadcast(val, src_lane):
    """Value of ``src_lane`` delivered to every lane.  One shuffle."""
    return cute.arch.shuffle_sync(val, cutlass.Int32(src_lane))


@cute.jit
def ballot_rank(predicate, lane_mask_lt):
    """Rank of this lane among the lanes whose predicate is true: the number of lower lanes
    with a true predicate.  Together with ``ballot_count`` this is the warp-aggregated
    ticket: one atomic per warp reserves ``count`` slots and each true lane takes
    ``base + rank``.  One ballot and one popcount.

    ``lane_mask_lt`` is ``cute.arch.lanemask_lt()`` (bits of lanes below this one), computed
    once per thread by the caller.
    """
    votes = cute.arch.vote_ballot_sync(predicate)
    return cutlass.Int32(cute.arch.popc(votes & lane_mask_lt))


@cute.jit
def ballot_count(predicate):
    """Number of lanes in the warp whose predicate is true.  One ballot and one popcount."""
    return cutlass.Int32(cute.arch.popc(cute.arch.vote_ballot_sync(predicate)))


@dsl_user_op
def match_any(value: cutlass.Int32, *, loc=None, ip=None):
    """Mask of the lanes holding the same 32-bit ``value`` as this lane (``match.any.sync``).

    All 32 lanes must participate.  One instruction (SM70+), but a slow one (~16 issue
    cycles).  The building block of warp-aggregated histogram increments: the lowest lane of
    each group adds the group's size once, so low-entropy inputs (many equal bins) send one
    atomic per warp instead of one per element.
    """
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [cutlass.Int32(value).ir_value(loc=loc, ip=ip)],
            "match.any.sync.b32 $0, $1, 0xffffffff;",
            "=r,r",
            has_side_effects=True,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )
