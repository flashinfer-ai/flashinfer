"""Filter pass: one streaming pass over the CTA's slice keeping everything above the threshold.

Structure (gvr_2's, adopted after two measured dead ends: unconditional per-element side
effects were 1.4-3x slower at 3-6% survivor rates, and a monolithic predicated-asm emit lost
ptxas's scheduling and ran 20% behind):

* Each iteration a thread loads ``unroll`` 16-byte vectors ``threads`` vectors apart and
  classifies every element into one bit of a dead mask: one compare, no memory effect.  For
  16-bit rows the packed compare handles eight elements in four instructions.
* The survivors of the warp are counted with a popcount and staged with one
  ``warp_reserve`` per warp per iteration.
* A divergent bit-walk over the survivors re-loads each one with a scalar load (holding all
  the loaded values live across the walk spills; the reload hits L1), writes key bits and
  index to the stage, and counts its survivor bin.  Overflow beyond the stage capacity goes
  to a trash slot; the count still grows, so the caller sees the overflow.
* The trip count is CTA-uniform (full iterations unguarded, the boundary iteration clamps
  addresses and masks validity) so the warp collectives are always converged.

Loop invariants are pinned (``device.compiler``) so NVVM does not rematerialize their defining
chains inside the loop at the 64-register wall.  Measured: an L2 prefetch of the next tile
made this pass slower (issue-bound, ~1 us per 64 KB tile per SM regardless of CTA count).
"""

import cutlass
import cutlass.cute as cute

from ...block.reduce import warp_reserve
from ...device.atomics import shared_add, shared_count
from ...device.compiler import pin_i32, pin_i64, pin_shared_address
from ...device.keys import below_or_equal_mask_16x8, pack_threshold_16
from ...device.memory import load_global_readonly_16, load_shared_8, store_shared_8

from .binning import survivor_bin

__all__ = ["filter_pass", "stage_pair"]


@cute.jit
def _classify_vector(
    elems,
    dead,
    w0,
    w1,
    w2,
    w3,
    bar,
    packed_bar,
    base_bit: cutlass.Constexpr,
    packed: cutlass.Constexpr,
):
    """OR into ``dead`` one bit per element of the vector that is <= bar (NaN never dies)."""
    if cutlass.const_expr(packed):
        dead = below_or_equal_mask_16x8(
            dead, w0, w1, w2, w3, packed_bar, base_bit, elems.is_bf16
        )
    else:
        for j in cutlass.range_constexpr(4):
            for h in cutlass.range_constexpr(elems.per_word):
                v = elems.value(elems.bits((w0, w1, w2, w3)[j], h))
                dead = dead | (
                    cutlass.Int32(v <= bar) << (base_bit + j * elems.per_word + h)
                )
    return dead


@cute.jit
def _stage_bits(
    elems,
    bits,
    idx,
    pos,
    capacity: cutlass.Constexpr,
    bar,
    scale,
    hist_base,
    stage_base,
):
    """Write one loaded survivor to the stage (slot ``capacity`` is the overflow trash) and count its bin.

    The stage is an array of (bits, index) pairs, 8 bytes per slot, written with one
    ``st.shared.v2`` (v0.1.31; two 32-bit stores into separate key and index arrays before:
    the ncu instruction profile against gvr_2 had our filter at 130K STS against its 60K).
    """
    slot = pos
    if slot > cutlass.Int32(capacity):
        slot = cutlass.Int32(capacity)
    store_shared_8(stage_base + slot * 8, bits, cutlass.Uint32(idx))
    shared_count(hist_base + survivor_bin(elems.value(bits), bar, scale) * 4)


@cute.jit
def _stage(
    elems,
    row_ptr,
    idx,
    pos,
    capacity: cutlass.Constexpr,
    bar,
    scale,
    hist_base,
    stage_base,
):
    """Reload one survivor and stage it."""
    _stage_bits(
        elems,
        elems.load_bits(row_ptr, idx),
        idx,
        pos,
        capacity,
        bar,
        scale,
        hist_base,
        stage_base,
    )


@cute.jit
def stage_pair(stage_base, t):
    """(bits, index) of stage slot ``t``: one ``ld.shared.v2``."""
    low, high = load_shared_8(stage_base + t * 8)
    return cutlass.Uint32(low), cutlass.Int32(high)


@cute.jit
def _next_bit(
    alive,
    lg: cutlass.Constexpr,
    per_vector: cutlass.Constexpr,
    vbase,
    start,
    threads: cutlass.Constexpr,
):
    """(alive without its lowest bit, row index of that bit's element).  With ``alive == 0`` the
    index is that of bit 0 (a valid element of the tile), so a predicated-off reload is safe."""
    low = alive & (cutlass.Int32(0) - alive)
    bit = cutlass.Int32(cute.arch.popc(low - cutlass.Int32(1)))
    if alive == 0:
        bit = cutlass.Int32(0)
    vec = vbase + (bit >> cutlass.Int32(lg)) * cutlass.Int32(threads)
    idx = start + (vec << cutlass.Int32(lg)) + (bit & cutlass.Int32(per_vector - 1))
    return alive & (alive - cutlass.Int32(1)), idx


@cute.jit
def filter_pass(
    elems,
    row_ptr,
    start,
    count,
    bar,
    scale,
    packed: cutlass.Constexpr,
    capacity: cutlass.Constexpr,
    s_count,
    s_hist,
    s_stage,
    tidx,
    threads: cutlass.Constexpr,
    unroll: cutlass.Constexpr,
    walk_width: cutlass.Constexpr = 1,
):
    """Stage every element of ``row[start, start + count)`` that is not <= ``bar``.

    On exit (after the closing barrier): ``s_count[0]`` = survivor count (may exceed
    ``capacity``), ``s_stage`` slots ``[0, min(count, capacity))`` = (bits, index) pairs, 8
    bytes each (``stage_pair`` reads one), ``s_hist`` = 256-bin survivor histogram over ALL
    survivors (staged or not).  ``s_stage`` is an 8-byte aligned Int32 array of at least
    ``2 * (capacity + 1)`` words.  Two barriers (one to zero, one to publish).
    ``unroll * per_vector`` must be at most 32 (the mask width).

    ``walk_width``: survivors reloaded per walk step, their loads issued together before any
    is staged.  With one, each survivor's reload waits out the previous one's latency; when
    two CTAs share an SM the tiles no longer fit the L1 left beside their shared memory and
    the reloads come from L2, so at k=2048 (3000+ survivors per row) the chain was the
    k-proportional cost of the wide-batch cells (ncu: long-scoreboard stalls 6 per issue at 34%
    issue utilization).
    """
    if tidx == 0:
        s_count[0] = cutlass.Int32(0)
    if tidx < 256:
        s_hist[tidx] = cutlass.Int32(0)
    cute.arch.barrier()

    per_vector = cutlass.const_expr(elems.per_vector)
    lg = cutlass.const_expr(elems.log2_per_vector)
    per_iter = cutlass.const_expr(unroll * threads)  # vectors per iteration
    all_valid = cutlass.const_expr(
        -1 if unroll * per_vector == 32 else (1 << (unroll * per_vector)) - 1
    )
    lane = tidx % 32
    hist_base = pin_shared_address(s_hist.toint())
    stage_base = pin_shared_address(s_stage.toint())
    base = pin_i64(row_ptr.toint() + cutlass.Int64(start) * cutlass.Int64(elems.bytes))
    n_vectors = count >> cutlass.Int32(lg)
    last_vector = pin_i32(n_vectors - 1)
    n_iter = pin_i32(
        (n_vectors + cutlass.Int32(per_iter - 1)) // cutlass.Int32(per_iter)
    )
    n_full = pin_i32(n_vectors // cutlass.Int32(per_iter))
    stride = cutlass.Int64(threads * 16)
    packed_bar = cutlass.Uint32(0)
    if cutlass.const_expr(packed):
        packed_bar = pack_threshold_16(bar, elems.is_bf16)

    it = cutlass.Int32(0)
    vbase = cutlass.Int32(tidx)  # first vector index of this thread this iteration
    vaddr = base + cutlass.Int64(tidx) * 16
    while it < n_iter:
        dead = cutlass.Int32(0)
        valid = cutlass.Int32(all_valid)
        if it < n_full:
            words = [load_global_readonly_16(vaddr + stride * u) for u in range(unroll)]
            for u in cutlass.range_constexpr(unroll):
                dead = _classify_vector(
                    elems, dead, *words[u], bar, packed_bar, u * per_vector, packed
                )
        else:  # boundary iteration: clamp addresses, mask validity
            valid = cutlass.Int32(0)
            for u in cutlass.range_constexpr(unroll):
                vi = vbase + cutlass.Int32(u * threads)
                clamped = vi
                if clamped > last_vector:
                    clamped = last_vector
                w0, w1, w2, w3 = load_global_readonly_16(
                    base + cutlass.Int64(clamped) * 16
                )
                if vi < n_vectors:
                    valid = valid | (
                        cutlass.Int32((1 << per_vector) - 1) << (u * per_vector)
                    )
                    dead = _classify_vector(
                        elems,
                        dead,
                        w0,
                        w1,
                        w2,
                        w3,
                        bar,
                        packed_bar,
                        u * per_vector,
                        packed,
                    )
        alive = (~dead) & valid
        pos = warp_reserve(cutlass.Int32(cute.arch.popc(alive)), lane, s_count)
        if cutlass.const_expr(walk_width == 1):
            while alive != 0:
                alive, idx = _next_bit(alive, lg, per_vector, vbase, start, threads)
                _stage(
                    elems,
                    row_ptr,
                    idx,
                    pos,
                    capacity,
                    bar,
                    scale,
                    hist_base,
                    stage_base,
                )
                pos = pos + cutlass.Int32(1)
        else:
            while alive != 0:
                # up to walk_width survivors: all their reloads first (independent loads in
                # flight), then the stage writes; absent survivors reload a valid element and
                # skip the write
                idxs = []
                present = []
                for _w in cutlass.range_constexpr(walk_width):
                    present.append(alive != 0)
                    alive, idx_w = _next_bit(
                        alive, lg, per_vector, vbase, start, threads
                    )
                    idxs.append(idx_w)
                loaded = [elems.load_bits(row_ptr, idxs[w]) for w in range(walk_width)]
                for w in cutlass.range_constexpr(walk_width):
                    if present[w]:
                        _stage_bits(
                            elems,
                            loaded[w],
                            idxs[w],
                            pos,
                            capacity,
                            bar,
                            scale,
                            hist_base,
                            stage_base,
                        )
                        pos = pos + cutlass.Int32(1)
        vbase = vbase + cutlass.Int32(per_iter)
        vaddr = vaddr + stride * unroll
        it = it + cutlass.Int32(1)

    # elements past the last full vector (fewer than per_vector of them)
    tail_start = n_vectors << cutlass.Int32(lg)
    for i in range(tidx, count - tail_start, threads):
        idx = start + tail_start + i
        bits = elems.load_bits(row_ptr, idx)
        if not (elems.value(bits) <= bar):
            pos = shared_add(s_count, 1)
            _stage_bits(
                elems, bits, idx, pos, capacity, bar, scale, hist_base, stage_base
            )
    cute.arch.barrier()
