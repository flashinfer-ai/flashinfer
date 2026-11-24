# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Queued native BF16 register selection, derived from TensorRT-LLM GVR V2."""

import cutlass

import cutlass.cute as cute

from cutlass._mlir.dialects import llvm

from cutlass.cutlass_dsl import T as dsl_types


from cutlass.cutlass_dsl import dsl_user_op

from . import packed_helpers as gvr
from ...pdl import pdl_prologue

RES_B = 0

RES_M = 1

RES_ABOVE = 2

RES_TOT = 3


@dsl_user_op
def _nan_to_negative_infinity(
    packed_pair: cutlass.Uint32, *, loc=None, ip=None
) -> cutlass.Uint32:
    """Map NaN halves to negative infinity before ordered-key selection."""
    return cutlass.Uint32(
        llvm.inline_asm(
            dsl_types.i32(),
            [packed_pair.ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b32 nan_mask;\n\t"
            "set.nan.u32.bf16x2 nan_mask, $1, $1;\n\t"
            "and.b32 $0, nan_mask, 0xff80ff80;\n\t"
            "not.b32 nan_mask, nan_mask;\n\t"
            "and.b32 nan_mask, $1, nan_mask;\n\t"
            "or.b32 $0, $0, nan_mask;\n}",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _sign_quad(
    lower_pair: cutlass.Uint32, upper_pair: cutlass.Uint32, *, loc=None, ip=None
) -> cutlass.Uint32:
    """Returns the sign bits of the four BF16 values in two packed pairs.

    Value ``i`` is in bit 7 of byte ``i``; the other bits are 0.
    """
    return cutlass.Uint32(
        llvm.inline_asm(
            dsl_types.i32(),
            [lower_pair.ir_value(loc=loc, ip=ip), upper_pair.ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b32 signs;\n\t"
            "prmt.b32 signs, $1, $2, 0x7531;\n\t"
            "and.b32 $0, signs, 0x80808080;\n}",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ordered_key_pair(
    packed_pair: cutlass.Uint32, *, loc=None, ip=None
) -> cutlass.Uint32:
    """Returns the ordered 16-bit keys of both halves of a packed 16-bit float pair.

    Negative halves flip all bits and the others flip the sign bit, so keys compare as unsigned in
    value order. The all-ones padding word maps to key 0, below every non-NaN value.
    """
    return cutlass.Uint32(
        llvm.inline_asm(
            dsl_types.i32(),
            [packed_pair.ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b32 sign_mask;\n\t"
            "prmt.b32 sign_mask, $1, 0, 0xbb99;\n\t"
            "lop3.b32 $0, $1, 0x80008000, sign_mask, 0x1e;\n}",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _half_code(
    packed_pair: cutlass.Uint32,
    upper: bool,
    signed: bool,
    *,
    loc=None,
    ip=None,
) -> cutlass.Int32:
    """Returns the ``upper`` or lower 16-bit half of ``packed_pair``.

    The half is zero-extended, or sign-extended with ``signed``.
    """
    if signed:
        selector = "0xbb32" if upper else "0x9910"
    else:
        selector = "0x4432" if upper else "0x4410"
    return cutlass.Int32(
        llvm.inline_asm(
            dsl_types.i32(),
            [packed_pair.ir_value(loc=loc, ip=ip)],
            f"prmt.b32 $0, $1, 0, {selector};",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _code_radix_cursor(
    key, crossing_high, threshold, high_address, low_address, tie_address
):
    """Returns ``key``'s output slot in the two-pass 8-bit code radix.

    The result is 0x40000000 when ``key`` is below ``threshold``. The slot is an atomic increment of
    the cursor of the key's high byte, of its low byte when the high byte is ``crossing_high``, or
    of the tie counter at ``tie_address`` when it equals ``threshold``.
    """
    high_byte = key >> cutlass.Uint32(8)
    cursor_address = high_address + cutlass.Int32(high_byte << cutlass.Uint32(2))
    if high_byte == crossing_high:
        cursor_address = low_address + cutlass.Int32(
            (key & cutlass.Uint32(255)) << cutlass.Uint32(2)
        )
    if key == threshold:
        cursor_address = tie_address
    return gvr._atom_shared_add1_if_ge(cursor_address, key, threshold)


@dsl_user_op
def _packed_clamp_bins(
    packed_pair: cutlass.Uint32,
    flip: cutlass.Uint32,
    lower: cutlass.Uint32,
    upper: cutlass.Uint32,
    pivot: cutlass.Uint32,
    positive: bool,
    *,
    loc=None,
    ip=None,
) -> cutlass.Uint32:
    """Returns the window bins of both halves of a packed BF16 pair, still packed.

    Each code is clamped to this lane's edges ``[lower, upper]`` and offset by ``pivot``, all
    replicated per half (see ``_window_controls``). The negative window first inverts the codes
    (``flip`` is all ones); the ``positive`` window clamps as signed 16-bit, so negative values land
    on the lower edge.
    """
    operands = (packed_pair, flip, lower, upper, pivot)
    instructions = (
        "xor.b32 code_pair, $1, $2;\n\t"
        "max.u16x2 code_pair, code_pair, $3;\n\t"
        "min.u16x2 code_pair, code_pair, $4;\n\t"
        "sub.u32 $0, code_pair, $5;\n}"
    )
    constraints = "=r,r,r,r,r,r"
    if positive:
        operands = (packed_pair, lower, upper, pivot)
        instructions = (
            "max.s16x2 code_pair, $1, $2;\n\t"
            "min.s16x2 code_pair, code_pair, $3;\n\t"
            "sub.u32 $0, code_pair, $4;\n}"
        )
        constraints = "=r,r,r,r,r"
    return cutlass.Uint32(
        llvm.inline_asm(
            dsl_types.i32(),
            [value.ir_value(loc=loc, ip=ip) for value in operands],
            "{\n\t.reg .b32 code_pair;\n\t" + instructions,
            constraints,
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _packed_clamp_cursor(
    histogram_address: cutlass.Int32,
    bin_index: cutlass.Int32,
    crossing_bin: cutlass.Int32,
    *,
    loc=None,
    ip=None,
) -> cutlass.Int32:
    """Returns window bin ``bin_index``'s next output slot, or 0x40000000 below ``crossing_bin``.

    The slot is an atomic increment of the bin's cursor in the histogram at ``histogram_address``.
    """
    return cutlass.Int32(
        llvm.inline_asm(
            dsl_types.i32(),
            [
                value.ir_value(loc=loc, ip=ip)
                for value in (histogram_address, bin_index, crossing_bin)
            ],
            "{\n\t.reg .b32 address;\n\t.reg .pred selected;\n\t"
            "setp.ge.u32 selected, $2, $3;\n\t"
            "mad.lo.u32 address, $2, 4, $1;\n\t"
            "mov.u32 $0, 0x40000000;\n\t"
            "@selected atom.relaxed.cta.shared.add.u32 $0, [address], 1;\n}",
            "=r,r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


class GvrPackedClampWindowKernel:
    """Select BF16 TopK using a sampled histogram and exact radix selection at the cutoff.
    NaNs map to -inf, ties are unordered and short rows are padded with -1.
    """

    def __init__(self, *, pdl: bool = False) -> None:
        self.pdl = bool(pdl)
        self.blk = 512
        self.vpt = 8
        self.nbh = 1024
        self.minb = 1
        self.parts = 1
        self.sample_warp = 4
        self.dtype = cutlass.BFloat16
        self.capacity = self.blk * (4 * self.vpt + 1)
        self.histogram_words = self.nbh
        self.output_offset_words = gvr.STATIC_WORDS + self.histogram_words
        self.required_smem_bytes = self.output_offset_words * 4

    def _histogram_offset(self, scalar_fragment: int) -> int:
        """Locate a scalar fragment in the single shared histogram."""
        return (scalar_fragment * self.parts // (4 * self.vpt)) * self.nbh * 4

    @cute.jit
    def _window_controls(self, window_base, thread_index, positive: cutlass.Constexpr):
        """Return packed flip, lower, upper and pivot controls for code-space bins.

        Clamp the base to keep the window in range. Lane-staggered edges
        distribute out-of-window codes across 32 bins on each side.
        """
        code_base = window_base
        flip = cutlass.Uint32(0x80008000)
        maximum_base = cutlass.Int32(32768 - self.nbh)
        if cutlass.const_expr(not positive):
            code_base = cutlass.Int32(0xFFFF) - window_base
            flip = cutlass.Uint32(0xFFFFFFFF)
            maximum_base = cutlass.Int32(65536 - self.nbh)
        if code_base < cutlass.Int32(0):
            code_base = cutlass.Int32(0)
        if code_base > maximum_base:
            code_base = maximum_base
        lane_index = thread_index & cutlass.Int32(31)
        pivot = cutlass.Uint32(code_base) * cutlass.Uint32(65537)
        lower = cutlass.Uint32(code_base + lane_index) * cutlass.Uint32(65537)
        upper = cutlass.Uint32(
            code_base + cutlass.Int32(self.nbh - 32) + lane_index
        ) * cutlass.Uint32(65537)
        return flip, lower, upper, pivot

    @cute.jit
    def _fill_window_histogram(
        self,
        packed_fragments,
        raw_tail,
        histogram_address,
        window_base,
        thread_index,
        body_vectors,
        tail_count,
        positive: cutlass.Constexpr,
        full: cutlass.Constexpr = False,
    ):
        """Replaces valid ``packed_fragments`` pairs by their window bins and counts every value.

        Values count into their part's sub-histogram (``_histogram_offset``), the tail value into
        the last part's. Returns the tail value's bin, or 0 for threads without one.
        """
        flip, lower, upper, pivot = self._window_controls(
            window_base, thread_index, positive
        )
        for fragment_index in cutlass.range_constexpr(self.vpt):
            vector_index = thread_index + cutlass.Int32(fragment_index * self.blk)
            if cutlass.const_expr(full) or vector_index < body_vectors:
                for pair_index in cutlass.range_constexpr(2):
                    bins = _packed_clamp_bins(
                        packed_fragments[fragment_index][pair_index],
                        flip,
                        lower,
                        upper,
                        pivot,
                        positive,
                    )
                    packed_fragments[fragment_index][pair_index] = bins
                    for half_index in cutlass.range_constexpr(2):
                        bin_index = _half_code(bins, half_index == 1, False)
                        gvr._red_shared_add1__reg(
                            histogram_address
                            + cutlass.Int32(
                                self._histogram_offset(
                                    fragment_index * 4 + pair_index * 2 + half_index
                                )
                            )
                            + bin_index * cutlass.Int32(4)
                        )
        tail_bin = cutlass.Int32(0)
        if thread_index < tail_count:
            tail_bins = _packed_clamp_bins(
                raw_tail, flip, lower, upper, pivot, positive
            )
            tail_bin = _half_code(tail_bins, False, False)
            gvr._red_shared_add1__reg(
                histogram_address
                + cutlass.Int32((self.parts - 1) * self.nbh * 4)
                + tail_bin * cutlass.Int32(4)
            )
        return tail_bin

    @cute.jit
    def _emit_window(
        self,
        packed_fragments,
        tail_bin,
        output_row,
        thread_index,
        part_index,
        tail_index,
        histogram_address,
        window_base,
        crossing_bin,
        top_k,
        body_vectors,
        tail_count,
        positive: cutlass.Constexpr,
        full: cutlass.Constexpr = False,
    ):
        """Emit this CTA's value-slot group through the window cursors.

        The last part owns the tail; only slots below top_k are written.
        """
        for group_index in cutlass.range_constexpr(self.parts):
            if part_index == cutlass.Int32(group_index):
                for scalar_fragment in cutlass.range_constexpr(
                    group_index * 4 * self.vpt // self.parts,
                    (group_index + 1) * 4 * self.vpt // self.parts,
                ):
                    vector_index = thread_index + cutlass.Int32(
                        (scalar_fragment // 4) * self.blk
                    )
                    if cutlass.const_expr(full) or vector_index < body_vectors:
                        bins = packed_fragments[scalar_fragment // 4][
                            (scalar_fragment % 4) // 2
                        ]
                        bin_index = _half_code(bins, scalar_fragment % 2 == 1, False)
                        output_position = _packed_clamp_cursor(
                            histogram_address, bin_index, crossing_bin
                        )
                        if output_position < top_k:
                            selected_index = vector_index * cutlass.Int32(
                                4
                            ) + cutlass.Int32(scalar_fragment % 4)
                            output_row[output_position] = selected_index
                if cutlass.const_expr(group_index == self.parts - 1):
                    if thread_index < tail_count:
                        output_position = _packed_clamp_cursor(
                            histogram_address, tail_bin, crossing_bin
                        )
                        if output_position < top_k:
                            output_row[output_position] = tail_index

    @cute.jit
    def _row_body(
        self,
        packed_fragments,
        envelope_raw_tail,
        envelope_body_vectors,
        row_address,
        shared_base,
        shared_histogram,
        histogram_address,
        output_row,
        thread_index,
        part_index,
        lane_index,
        warp_index,
        top_k,
        row_length,
        body_vectors,
        full: cutlass.Constexpr,
    ):
        """Select a row through a sampled window, with exact radix fallback.

        Normalize NaNs and collect sample extrema. Emit directly for a
        single-code crossing bin; otherwise part 0 runs two-pass radix.
        full omits vector bounds checks.
        """
        tail_begin = body_vectors << cutlass.Int32(2)
        tail_count = row_length - tail_begin
        tail_index = tail_begin + thread_index
        local_minimum = cutlass.Uint32(0xFFFF)
        local_maximum = cutlass.Uint32(0)
        positive_local = cutlass.Int32(0)
        for fragment_index in cutlass.range_constexpr(self.vpt):
            vector_index = thread_index + cutlass.Int32(fragment_index * self.blk)
            if cutlass.const_expr(full) or vector_index < body_vectors:
                packed_fragments[fragment_index][0] = _nan_to_negative_infinity(
                    packed_fragments[fragment_index][0]
                )
                packed_fragments[fragment_index][1] = _nan_to_negative_infinity(
                    packed_fragments[fragment_index][1]
                )
                positive_local = (
                    positive_local
                    + cutlass.Int32(4)
                    - cutlass.Int32(
                        cute.arch.popc(
                            cutlass.Int32(
                                _sign_quad(
                                    packed_fragments[fragment_index][0],
                                    packed_fragments[fragment_index][1],
                                )
                            )
                        )
                    )
                )
            else:
                packed_fragments[fragment_index][0] = cutlass.Uint32(0xFFFFFFFF)
                packed_fragments[fragment_index][1] = cutlass.Uint32(0xFFFFFFFF)
        raw_tail = cutlass.Uint32(0xFFFF)
        if thread_index < tail_count:
            if body_vectors == envelope_body_vectors:
                raw_tail = envelope_raw_tail
            else:
                raw_tail = gvr._ld_g_nc_u16(
                    row_address + cutlass.Int64(tail_index) * cutlass.Int64(2)
                )
            raw_tail = _nan_to_negative_infinity(raw_tail)
            positive_local = positive_local + cutlass.Int32(
                (raw_tail & cutlass.Uint32(0x8000)) == cutlass.Uint32(0)
            )

        shared_result = gvr._smem_view(cutlass.Int32, shared_base, 0, 6)
        shared_count = gvr._smem_view(cutlass.Int32, shared_base, 6, 2, align=4)
        shared_work = gvr._smem_view(cutlass.Int32, shared_base, 16, 32)
        positive_local = cutlass.Int32(cute.arch.warp_redux_sync(positive_local, "add"))
        if lane_index == cutlass.Int32(0):
            shared_work[warp_index] = positive_local
        shared_samples = gvr._smem_view(cutlass.Uint32, shared_base, 48, 3)
        if warp_index == cutlass.Int32(self.sample_warp):
            local_minimum = cutlass.Uint32(0xFFFF)
            local_maximum = cutlass.Uint32(0)
            for fragment_index in cutlass.range_constexpr(self.vpt):
                vector_index = thread_index + cutlass.Int32(fragment_index * self.blk)
                if cutlass.const_expr(full) or vector_index < body_vectors:
                    sample_pair = _ordered_key_pair(packed_fragments[fragment_index][1])
                    sample_key = cutlass.Uint32(_half_code(sample_pair, True, False))
                    if sample_key < local_minimum:
                        local_minimum = sample_key
                    if sample_key > local_maximum:
                        local_maximum = sample_key
            local_minimum = cute.arch.warp_redux_sync(local_minimum, "min")
            local_maximum = cute.arch.warp_redux_sync(local_maximum, "max")
            if lane_index == cutlass.Int32(0):
                shared_samples[0] = local_minimum
                shared_samples[1] = local_maximum
        cute.arch.barrier()
        sample_minimum = cutlass.Uint32(shared_samples[0])
        sample_maximum = cutlass.Uint32(shared_samples[1])
        positive_count = cutlass.Int32(0)
        if lane_index < cutlass.Int32(self.blk // 32):
            positive_count = shared_work[lane_index]
        positive_count = cutlass.Int32(cute.arch.warp_redux_sync(positive_count, "add"))
        negative_cut = positive_count < top_k
        window_base = (
            cutlass.Int32(sample_maximum)
            - cutlass.Int32(0x8000)
            - cutlass.Int32(self.nbh - 33)
        )
        if negative_cut:
            window_base = cutlass.Int32(
                sample_minimum ^ cutlass.Uint32(0xFFFF)
            ) + cutlass.Int32(1)
        elif window_base < cutlass.Int32(0):
            window_base = cutlass.Int32(0)
        tail_bin = cutlass.Int32(0)
        if negative_cut:
            tail_bin = self._fill_window_histogram(
                packed_fragments,
                raw_tail,
                histogram_address,
                window_base,
                thread_index,
                body_vectors,
                tail_count,
                positive=False,
                full=full,
            )
        else:
            tail_bin = self._fill_window_histogram(
                packed_fragments,
                raw_tail,
                histogram_address,
                window_base,
                thread_index,
                body_vectors,
                tail_count,
                positive=True,
                full=full,
            )
        cute.arch.barrier()
        gvr.scan_cross_w(
            shared_histogram,
            shared_work,
            top_k,
            thread_index,
            shared_result,
            blk=self.blk,
            nb=self.nbh,
        )
        cute.arch.barrier()
        crossing_bin = shared_result[gvr.RES_B]
        if (
            shared_result[gvr.RES_TOT] >= top_k
            and shared_result[gvr.RES_M] > cutlass.Int32(0)
            and crossing_bin >= cutlass.Int32(32)
            and crossing_bin < cutlass.Int32(self.nbh - 32)
            and (
                not negative_cut or window_base - crossing_bin >= cutlass.Int32(0x8000)
            )
        ):
            if negative_cut:
                self._emit_window(
                    packed_fragments,
                    tail_bin,
                    output_row,
                    thread_index,
                    part_index,
                    tail_index,
                    histogram_address,
                    window_base,
                    crossing_bin,
                    top_k,
                    body_vectors,
                    tail_count,
                    positive=False,
                    full=full,
                )
            else:
                self._emit_window(
                    packed_fragments,
                    tail_bin,
                    output_row,
                    thread_index,
                    part_index,
                    tail_index,
                    histogram_address,
                    window_base,
                    crossing_bin,
                    top_k,
                    body_vectors,
                    tail_count,
                    positive=True,
                    full=full,
                )
        elif part_index == cutlass.Int32(0):
            for fragment_index in cutlass.range_constexpr(self.vpt):
                vector_index = thread_index + cutlass.Int32(fragment_index * self.blk)
                if cutlass.const_expr(full) or vector_index < body_vectors:
                    fallback_lower, fallback_upper = gvr._ld_g_nc_v2_b32(
                        row_address + cutlass.Int64(vector_index) * cutlass.Int64(8)
                    )
                    packed_fragments[fragment_index][0] = _nan_to_negative_infinity(
                        fallback_lower
                    )
                    packed_fragments[fragment_index][1] = _nan_to_negative_infinity(
                        fallback_upper
                    )
            for fragment_index in cutlass.range_constexpr(self.vpt):
                for pair_index in cutlass.range_constexpr(2):
                    packed_fragments[fragment_index][pair_index] = _ordered_key_pair(
                        packed_fragments[fragment_index][pair_index]
                    )
            tail_key = cutlass.Uint32(0)
            if thread_index < tail_count:
                tail_key = cutlass.Uint32(gvr.fkey16_of_code16(raw_tail))
            shared_low = gvr._smem_view(
                cutlass.Int32, shared_base, gvr.STATIC_WORDS + 256, 256
            )
            low_address = histogram_address + cutlass.Int32(1024)
            for histogram_batch in cutlass.range_constexpr(
                (512 + self.blk - 1) // self.blk
            ):
                histogram_index = thread_index + cutlass.Int32(
                    histogram_batch * self.blk
                )
                if histogram_index < cutlass.Int32(512):
                    shared_histogram[histogram_index] = cutlass.Int32(0)
            cute.arch.barrier()
            for fragment_index in cutlass.range_constexpr(self.vpt):
                vector_index = thread_index + cutlass.Int32(fragment_index * self.blk)
                if cutlass.const_expr(full) or vector_index < body_vectors:
                    for pair_index in cutlass.range_constexpr(2):
                        key_pair = packed_fragments[fragment_index][pair_index]
                        for half_index in cutlass.range_constexpr(2):
                            key = cutlass.Uint32(
                                _half_code(key_pair, half_index == 1, False)
                            )
                            gvr._red_shared_add1__reg(
                                histogram_address
                                + cutlass.Int32(
                                    (key >> cutlass.Uint32(8)) << cutlass.Uint32(2)
                                )
                            )
            if thread_index < tail_count:
                gvr._red_shared_add1__reg(
                    histogram_address
                    + cutlass.Int32(
                        (tail_key >> cutlass.Uint32(8)) << cutlass.Uint32(2)
                    )
                )
            cute.arch.barrier()
            gvr.find_cross256_cursors(
                histogram_address,
                top_k,
                thread_index,
                shared_result,
                cutlass.Int32(0),
                shared_count,
                shared_low,
                cutlass.Int32(0),
                cancel_pad=False,
            )
            cute.arch.barrier()
            crossing_high = cutlass.Uint32(shared_result[gvr.RES_B])
            above_high = shared_result[gvr.RES_ABOVE]
            for fragment_index in cutlass.range_constexpr(self.vpt):
                vector_index = thread_index + cutlass.Int32(fragment_index * self.blk)
                if cutlass.const_expr(full) or vector_index < body_vectors:
                    for pair_index in cutlass.range_constexpr(2):
                        key_pair = packed_fragments[fragment_index][pair_index]
                        for half_index in cutlass.range_constexpr(2):
                            key = cutlass.Uint32(
                                _half_code(key_pair, half_index == 1, False)
                            )
                            gvr._red_shared_add1_if_eq(
                                low_address
                                + cutlass.Int32(
                                    (key & cutlass.Uint32(255)) << cutlass.Uint32(2)
                                ),
                                key >> cutlass.Uint32(8),
                                crossing_high,
                            )
            if thread_index < tail_count:
                gvr._red_shared_add1_if_eq(
                    low_address
                    + cutlass.Int32(
                        (tail_key & cutlass.Uint32(255)) << cutlass.Uint32(2)
                    ),
                    tail_key >> cutlass.Uint32(8),
                    crossing_high,
                )
            cute.arch.barrier()
            gvr.find_cross256_cursors(
                low_address,
                top_k - above_high,
                thread_index,
                shared_result,
                above_high,
                shared_count,
                shared_low,
                cutlass.Int32(0),
                cancel_pad=False,
            )
            cute.arch.barrier()
            threshold = (crossing_high << cutlass.Uint32(8)) | cutlass.Uint32(
                shared_result[gvr.RES_B]
            )
            tie_address = shared_base + cutlass.Int32(7 * 4)
            for fragment_index in cutlass.range_constexpr(self.vpt):
                for pair_index in cutlass.range_constexpr(2):
                    key_pair = packed_fragments[fragment_index][pair_index]
                    for half_index in cutlass.range_constexpr(2):
                        key = cutlass.Uint32(
                            _half_code(key_pair, half_index == 1, False)
                        )
                        output_position = _code_radix_cursor(
                            key,
                            crossing_high,
                            threshold,
                            histogram_address,
                            low_address,
                            tie_address,
                        )
                        if output_position < top_k:
                            selected_index = (
                                thread_index + cutlass.Int32(fragment_index * self.blk)
                            ) * cutlass.Int32(4) + cutlass.Int32(
                                pair_index * 2 + half_index
                            )
                            output_row[output_position] = selected_index
            output_position = _code_radix_cursor(
                tail_key,
                crossing_high,
                threshold,
                histogram_address,
                low_address,
                tie_address,
            )
            if output_position < top_k:
                output_row[output_position] = tail_index

    @cute.kernel
    def kern(
        self,
        logits: cute.Tensor,
        pre_idx: cute.Tensor,
        kv_lens: cute.Tensor,
        out: cute.Tensor,
        envelope: cutlass.Int32,
        queue: cute.Tensor,
        blocks: cute.Tensor,
        logical: cute.Tensor,
    ):
        """Load and select one recovery row.

        Short rows emit directly; longer rows use _row_body.
        envelope bounds the row length.
        """
        pdl_prologue(self.pdl)
        thread_index, _, _ = cute.arch.thread_idx()
        block_index, _, _ = cute.arch.block_idx()
        for task in cutlass.range(block_index, queue[0], self.sms):
            row_index = queue[task + 1]
            part_index = cutlass.Int32(0)
            lane_index = thread_index & cutlass.Int32(31)
            warp_index = thread_index >> cutlass.Int32(5)
            top_k = cutlass.Int32(pre_idx.shape[1])
            shared_pointer = cute.arch.get_dyn_smem(cutlass.Int32, alignment=16)
            shared_base = shared_pointer.toint()
            output_row = out[row_index, None]
            envelope_body_vectors = envelope >> cutlass.Int32(2)
            if envelope_body_vectors > cutlass.Int32(self.blk * self.vpt):
                envelope_body_vectors = cutlass.Int32(self.blk * self.vpt)
            envelope_tail_begin = envelope_body_vectors << cutlass.Int32(2)
            envelope_tail_count = envelope - envelope_tail_begin
            envelope_tail_index = envelope_tail_begin + thread_index
            row_address = logits[row_index, None].iterator.toint()
            packed_fragments = [
                cute.make_rmem_tensor((2,), cutlass.Uint32) for _ in range(self.vpt)
            ]
            for fragment_index in cutlass.range_constexpr(self.vpt):
                packed_fragments[fragment_index][0] = cutlass.Uint32(0xFFFFFFFF)
                packed_fragments[fragment_index][1] = cutlass.Uint32(0xFFFFFFFF)
                vector_index = thread_index + cutlass.Int32(fragment_index * self.blk)
                if vector_index < envelope_body_vectors:
                    lower_word, upper_word = gvr._ld_g_nc_v2_b32(
                        row_address + cutlass.Int64(vector_index) * cutlass.Int64(8)
                    )
                    packed_fragments[fragment_index][0] = lower_word
                    packed_fragments[fragment_index][1] = upper_word
            envelope_raw_tail = cutlass.Uint32(0xFFFF)
            if thread_index < envelope_tail_count:
                envelope_raw_tail = gvr._ld_g_nc_u16(
                    row_address + cutlass.Int64(envelope_tail_index) * cutlass.Int64(2)
                )
            shared_histogram = gvr._smem_view(
                cutlass.Int32, shared_base, gvr.STATIC_WORDS, self.histogram_words
            )
            histogram_address = gvr._smem_addr_reg__reg(
                shared_base + cutlass.Int32(gvr.STATIC_WORDS * 4)
            )
            for clear_batch in cutlass.range_constexpr(
                (self.parts * self.nbh // 4 + self.blk - 1) // self.blk
            ):
                clear_index = thread_index + cutlass.Int32(clear_batch * self.blk)
                if cutlass.const_expr(
                    (clear_batch + 1) * self.blk <= self.parts * self.nbh // 4
                ) or clear_index < cutlass.Int32(self.parts * self.nbh // 4):
                    gvr._sts_v4_u32(
                        histogram_address + clear_index * cutlass.Int32(16),
                        cutlass.Int32(0),
                        cutlass.Int32(0),
                        cutlass.Int32(0),
                        cutlass.Int32(0),
                    )
            row_length = envelope
            row_length = kv_lens[row_index]
            if row_length < cutlass.Int32(0):
                row_length = cutlass.Int32(0)
            if row_length > envelope:
                row_length = envelope
            if row_length <= top_k:
                if part_index == cutlass.Int32(0):
                    output_index = thread_index
                    while output_index < top_k:
                        selected_index = cutlass.Int32(-1)
                        if output_index < row_length:
                            selected_index = output_index
                        output_row[output_index] = selected_index
                        output_index = output_index + cutlass.Int32(self.blk)
            else:
                body_vectors = row_length >> cutlass.Int32(2)
                if body_vectors > cutlass.Int32(self.blk * self.vpt):
                    body_vectors = cutlass.Int32(self.blk * self.vpt)
                if body_vectors == cutlass.Int32(self.blk * self.vpt):
                    self._row_body(
                        packed_fragments,
                        envelope_raw_tail,
                        envelope_body_vectors,
                        row_address,
                        shared_base,
                        shared_histogram,
                        histogram_address,
                        output_row,
                        thread_index,
                        part_index,
                        lane_index,
                        warp_index,
                        top_k,
                        row_length,
                        body_vectors,
                        full=True,
                    )
                else:
                    self._row_body(
                        packed_fragments,
                        envelope_raw_tail,
                        envelope_body_vectors,
                        row_address,
                        shared_base,
                        shared_histogram,
                        histogram_address,
                        output_row,
                        thread_index,
                        part_index,
                        lane_index,
                        warp_index,
                        top_k,
                        row_length,
                        body_vectors,
                        full=False,
                    )

            cute.arch.barrier()
            for col in cutlass.range(thread_index, top_k, self.blk):
                pick = out[row_index, col]
                value = cutlass.Int32(-1)
                if pick >= 0:
                    value = blocks[row_index, pick // 8] * 8 + pick % 8
                logical[row_index, col] = value

    @cute.jit
    def __call__(
        self, scores, ends, counts, blocks, out, logical, thresholds, queue, stream
    ):
        self.kern(
            scores,
            out,
            ends,
            out,
            cutlass.Int32(scores.shape[1]),
            queue,
            blocks,
            logical,
        ).launch(
            grid=(self.sms, 1, 1),
            block=(self.blk, 1, 1),
            stream=stream,
            smem=self.required_smem_bytes,
            min_blocks_per_mp=self.minb,
            use_pdl=self.pdl,
        )
