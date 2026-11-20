# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""The ONE visible-set helper of the SM107 index-list sparse forwards -- shared by the tcgen05 tile (``sparse_d256_f16.py``) and the
lean ``mma.sync`` decode tile (``sparse_lean_d256_f16.py``).

Not a template: nothing here reads ``FROST_TEMPLATE_PARAMS``.  Every config knob the arithmetic depends on is a ``cutlass.Constexpr``
parameter the calling template passes from ITS ``CFG`` (a constexpr parameter folds exactly like a module-global constant: the tcgen05
module's cubins are byte-identical before and after this lift, the md5 record in ``test_sdpa_fwd_sparse_lean_sm107.py``), so the two
kernel bodies cannot drift apart on what a row may see (sdpa-invariants: "one helper for the visible set").

WHAT A ROW SEES (the sparse tile's header, ``sparse_d256_f16.py``): for query row ``tok`` of sequence ``batch`` at key-space position
``pos`` with ``n_vis = clamp(min(pos + 1, L_b), 0)`` visible tokens and the list row ``item_row`` (``[.., block_topk]`` int32, a valid prefix
then -1, blocks of ``block_size`` tokens):

    pos           = tok  (top-left)  |  tok + (L_b - S_q_b)  (bottom_right: the row sits at the END of the sequence)
    anchor        = pos  |  L_b - S_q_b = pos_0, the step-0 position  (list_per_sequence: ONE list per sequence)
    n_sel_default = min(block_topk, floor(clamp(anchor + 1, 0) / block_size))
    count         = clamp(block_lens[item_row], 0, n_sel_default)  when given, else n_sel_default      (device values, never a host read)
    tail_lo       = floor(n_anchor / block_size)                   the first block NOT complete at the anchor
    n_tail        = max(0, ceil(n_vis / block_size) - tail_lo)     0 / 1 (anchor = pos), 0..2 under the shared list at S_q <= 4;
                                                                   0 when include_open_block = 0 (the pure-list test arm)
    dead          = (pos < 0) | (padded row) | (L_b <= 0) | (count + n_tail == 0)
    blk(idx)      = ids[item_row, idx]  for idx < count;  tail_lo + (idx - count)  for idx in [count, count + n_tail);  -1 past that;
                    -1 for every index of a dead item, a negative id, or an id above (2^31 - 1) / block_size (a large id must not
                    wrap into a live key once multiplied by block_size)

The tile / chunk cut (``n_tiles``, the split chunk) is NOT here: it is each body's own geometry (the tcgen05 tile's 128-key tiles and
``SPLIT_KV`` chunks in its ``_item_bounds``; the lean tile's 32-key sub-tiles and ``SUB_TILES_PER_CTA`` chunks in its own body).
"""

from typing import Optional

from cutlass._mlir.dialects import arith

import cutlass
import cutlass.cute as cute

__all__ = ["_select_f32", "_select_i32", "block_id_at", "visible_bounds"]


@cute.jit
def _select_i32(cond, a, b):
    return cutlass.Int32(arith.select(cond.ir_value(), cutlass.Int32(a).ir_value(), cutlass.Int32(b).ir_value()))


@cute.jit
def _select_f32(cond, a, b):
    return cutlass.Float32(arith.select(cond.ir_value(), a.ir_value(), b.ir_value()))


@cute.jit
def visible_bounds(
    tok,
    batch,
    item_row,
    eff_seqlen_q,
    block_lens_tensor: Optional[cute.Tensor],
    seq_kv_lens_tensor,
    seqlen_kv,
    *,
    seq_kv_lens_present: cutlass.Constexpr[int],
    bottom_right: cutlass.Constexpr[int],
    list_per_sequence: cutlass.Constexpr[int],
    include_open_block: cutlass.Constexpr[int],
    block_size: cutlass.Constexpr[int],
    block_topk: cutlass.Constexpr[int],
):
    """The row's key-space position, the entries read from the item's list, the appended tail block(s), the dead flag, the
    listed-block total, the sequence's KV length and the visible-token count -- ``(pos, count, n_tail, tail_lo, dead, n_listed,
    eff_seqlen_kv, n_vis)`` -- from the SAME inputs in every role and every body (the module docstring's arithmetic; ``n_vis`` is
    returned so a body's per-key mask derives from the helper's value, never from a copy of its line).  Every knob is a constexpr parameter so each arm
    folds at trace time (the dense rendering traces exactly the prefill form's program)."""
    eff_seqlen_kv = seqlen_kv
    if cutlass.const_expr(seq_kv_lens_present == 1):
        eff_seqlen_kv = cutlass.Int32(cutlass.make_array_view(seq_kv_lens_tensor)[batch])
    zero = cutlass.Int32(0)
    one = cutlass.Int32(1)
    bs = cutlass.Int32(block_size)
    # Every arm below is a const_expr fold spelled so the dense rendering (no arm) traces EXACTLY the prefill form's program.
    pos = tok
    if cutlass.const_expr(bottom_right == 1):
        pos = tok + (eff_seqlen_kv - eff_seqlen_q)
    n_vis = cute.math.max(cute.math.min(pos + one, eff_seqlen_kv), zero)
    anchor = pos
    n_anchor = n_vis
    if cutlass.const_expr(list_per_sequence == 1):
        anchor = eff_seqlen_kv - eff_seqlen_q  # pos_0: the step-0 position (BOTTOM_RIGHT is forced by the config)
        n_anchor = cute.math.max(cute.math.min(anchor + one, eff_seqlen_kv), zero)
    n_sel_default = cute.math.min(cutlass.Int32(block_topk), cute.math.max(anchor + one, zero) // bs)
    count = n_sel_default
    if cutlass.const_expr(block_lens_tensor is not None):
        given = cutlass.Int32(cutlass.make_array_view(block_lens_tensor)[item_row])
        count = cute.math.min(cute.math.max(given, zero), n_sel_default)
    tail_lo = n_anchor // bs
    n_tail = zero
    if cutlass.const_expr(include_open_block == 1):
        if cutlass.const_expr(list_per_sequence == 1):
            n_tail = cute.math.max((n_vis + bs - one) // bs - tail_lo, zero)
        else:
            # anchor = pos: the same count spelled as the prefill form's has_open
            n_tail = _select_i32((n_vis % bs) != zero, 1, 0)
    n_listed = count + n_tail
    dead = (pos < zero) | (pos >= eff_seqlen_q) | (eff_seqlen_kv <= zero) | (n_listed == zero)
    if cutlass.const_expr(bottom_right == 1):
        # the padded-row term is on the Q ROW, not the key-space position (identical when the diagonal is top-left)
        dead = (pos < zero) | (tok >= eff_seqlen_q) | (eff_seqlen_kv <= zero) | (n_listed == zero)
    return pos, count, n_tail, tail_lo, dead, n_listed, eff_seqlen_kv, n_vis


@cute.jit
def block_id_at(idx, loaded, count, n_tail, tail_lo, dead, *, list_per_sequence: cutlass.Constexpr[int], block_size: cutlass.Constexpr[int]):
    """The block id at list index ``idx`` of the current item: the loaded word while ``idx < count``, the tail blocks
    ``tail_lo .. tail_lo + n_tail - 1`` at ``idx == count .. count + n_tail - 1`` (0, 1 or 2 of them), -1 past that -- and -1
    for every index of a dead item.  ``loaded`` is the word the caller read unconditionally at an in-bounds address; this
    SELECT decides whether it means anything."""
    if cutlass.const_expr(list_per_sequence == 1):
        rel = idx - count
        tail = _select_i32((rel >= cutlass.Int32(0)) & (rel < n_tail), tail_lo + rel, cutlass.Int32(-1))
    else:
        # one tail id at most (n_tail in {0, 1}): the prefill form's spelling, so the dense rendering traces its own program
        tail = _select_i32((idx == count) & (n_tail != cutlass.Int32(0)), tail_lo, cutlass.Int32(-1))
    blk = _select_i32(idx < count, loaded, tail)
    # Bound the block domain BEFORE forming block_size * blk + lane: a large positive Int32 id must not wrap into a live key.
    invalid = dead | (blk < cutlass.Int32(0)) | (blk > cutlass.Int32((2**31 - 1) // block_size))
    return _select_i32(invalid, cutlass.Int32(-1), blk)
