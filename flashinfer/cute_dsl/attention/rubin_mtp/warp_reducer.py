# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Fixed-split native MTP reducer with four independent rows per CTA."""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32


class WarpMLAReducer:
    """Reduce one D512 row per warp, without a CTA-wide shared-weight barrier.

    The producer uses M256 workspace tiles and base-2 partial LSE. Output
    tensors describe the logical (head, column, query, batch) rows, including
    partial M256 tiles. The caller supplies a split budget in [1, max_splits].
    Empty rows produce zero output and negative-infinite LSE.
    """

    def __init__(self, max_splits: int = 32):
        if type(max_splits) is not int or not 1 <= max_splits <= 32:
            raise ValueError("warp reducer max_splits must be an integer in [1,32]")
        self.capacity = max_splits
        self.prefetch = 1
        self.warps = 4
        self.row_threads = 32
        self.rows_per_cta = 4
        self.bands = 4

    @cute.jit
    def __call__(
        self,
        out,
        lse,
        partials,
        partial_lse,
        splits,
        lengths,
        cum_seq_lens_q,
        block_split_kvs,
        scale,
        stream,
    ):
        if cutlass.const_expr(
            cum_seq_lens_q is not None or block_split_kvs is not None
        ):
            raise ValueError("warp reducer requires fixed query and split budgets")
        self.kernel(out, lse, partials, partial_lse, splits, lengths, scale).launch(
            grid=(
                cute.ceil_div(out.shape[0] * out.shape[2], self.rows_per_cta),
                lengths.shape[0],
                1,
            ),
            block=(32 * self.warps, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        out: cute.Tensor,
        lse: cute.Tensor,
        partials: cute.Tensor,
        partial_lse: cute.Tensor,
        splits: Int32,
        lengths: cute.Tensor,
        scale: Float32,
    ):
        bx, batch, _ = cute.arch.block_idx()
        tx, _, _ = cute.arch.thread_idx()
        lane = tx % 32
        row_group = cute.arch.make_warp_uniform(tx // self.row_threads)
        row_lane = tx % self.row_threads
        row = bx * self.rows_per_cta + row_group
        if row < out.shape[0] * out.shape[2]:
            head = row % out.shape[0]
            query = row // out.shape[0]
            tile, tile_row = row // 256, row % 256
            ktiles = cute.ceil_div(lengths[batch], 128)
            per = max(1, cute.ceil_div(ktiles, splits))
            active = cute.ceil_div(ktiles, per)
            # Each lane owns one split weight; each warp owns one output row.
            score = -Float32.inf
            if lane < active:
                score = partial_lse[tile_row, lane, tile, batch]
            largest = cute.arch.warp_reduction_max(score)
            safe = largest if largest != -Float32.inf else Float32(0)
            weight = cute.math.exp2(score - safe, fastmath=True)
            denom = cute.arch.warp_reduction_sum(weight)
            global_lse = -Float32.inf
            if denom > 0:
                global_lse = safe + cute.math.log2(denom, fastmath=True)
                weight = weight / denom
            if row_lane == 0:
                lse[head, query, batch] = global_lse * scale

            acc = cute.make_rmem_tensor(cute.make_layout((4, self.bands)), Float32)
            acc.fill(0)
            gp = partials[tile_row, None, None, tile, batch]
            atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128
            )
            thr = cute.make_tiled_copy_tv(
                atom,
                cute.make_ordered_layout((1, self.row_threads), order=(1, 0)),
                cute.make_layout((1, 4)),
            ).get_slice(row_lane)
            for group in cutlass.range_constexpr(
                cute.ceil_div(self.capacity, self.prefetch)
            ):
                first = group * self.prefetch
                if first < active:
                    src = cute.make_tensor(
                        gp.iterator + first * gp.stride[0],
                        cute.make_layout(
                            (self.prefetch, 512), stride=(gp.stride[0], 1)
                        ),
                    )
                    src_thread = thr.partition_S(src)
                    fragment = cute.make_fragment_like(src_thread)
                    predicate = cute.make_rmem_tensor(
                        cute.make_layout((1, self.prefetch, self.bands)),
                        cutlass.Boolean,
                    )
                    for s in cutlass.range_constexpr(self.prefetch):
                        for d in cutlass.range_constexpr(self.bands):
                            predicate[0, s, d] = first + s < active
                    fragment.fill(0)
                    cute.copy(atom, src_thread, fragment, pred=predicate)
                    for s in cutlass.range_constexpr(self.prefetch):
                        w = cute.arch.shuffle_sync(weight, first + s)
                        for d in cutlass.range_constexpr(self.bands):
                            for v in cutlass.range_constexpr(4):
                                p = fragment[v, s, d] if w != 0 else Float32(0)
                                acc[v, d] += p * w
            for d in cutlass.range_constexpr(self.bands):
                for v in cutlass.range_constexpr(4):
                    out[
                        head,
                        d * (4 * self.row_threads) + row_lane * 4 + v,
                        query,
                        batch,
                    ] = acc[v, d].to(out.element_type)
