"""SMEM to GMEM store for the sm_80 delta-rule prefill kernel.

The sm_90 path stores O with a bulk tensor copy issued by a dedicated store
warp: one thread starts the tile and the hardware walks the descriptor,
clamping at the tensor bound. Neither half survives here.

The copy does not, because ``CopyBulkTensorTileS2GOp`` and the non-tensor
``CopyBulkS2GOp`` are both sm_90+, so the store is an ordinary vectorized one
and the bound the descriptor carried becomes a predicate -- a row past this
sequence's end is simply not written. That also drops the spare tensor map the
sm_90 path keeps in global memory and rewrites whenever a packed sequence ends
mid-tile, along with the fence around mutating it.

The dedicated warp does not survive either, for a reason that has nothing to do
with stores: through the pipeline this kernel can build there is no way for one
warp to wait on another warp's cp.async -- see pipeline_sm80 for which mbarrier
operations the DSL refuses on this target -- so every thread gets the same work
and the store warp has no one left to receive tiles from. The store is block-wide now, which
is also why it needs no pipeline -- the barrier before it is what makes the
math warps' writes visible, and the barrier after is what frees the buffer.
"""

import cutlass
import cutlass.cute as cute

# O is bf16 with d contiguous on both sides, so a lane's run of `vec`
# elements is one access on each.
_ELEMENT_BITS = 16  # bf16
_ALIGN_BYTES = 16


class CollectiveStoreSm80:
    """Writes O tiles from shared memory with predicated vector stores."""

    def __init__(self, blk_q: int, d: int, num_threads: int):
        self.BLK_Q = blk_q
        self.D = d
        self.num_threads = num_threads
        # Elements a thread moves per store. O is bf16 and d is contiguous, so
        # eight of them is one 16 B access -- the widest a thread can do.
        # Eight bf16 is 16 B, the widest a lane can store. A d that does not
        # divide by it is not a shape this kernel is built for, and narrowing
        # the vector silently would leave the copy atom claiming a width the
        # data does not have -- so it is refused instead.
        self.vec = 8
        if d % self.vec != 0:
            raise ValueError(
                f"head_size {d} is not a multiple of {self.vec}; the sm_80 O "
                f"store has no narrower path"
            )
        self.lanes_per_row = d // self.vec
        if num_threads % self.lanes_per_row != 0:
            raise ValueError(
                f"store needs a thread count divisible by {self.lanes_per_row} "
                f"(d={d} / vec={self.vec}), got {num_threads}"
            )
        self.rows_at_a_time = num_threads // self.lanes_per_row

    @cute.jit
    def run(
        self,
        sO: cute.Tensor,
        gO: cute.Tensor,
        work_desc,
        blk: cutlass.Int32,
        num_q_heads: cutlass.Int32,
        num_v_heads: cutlass.Int32,
        tid: cutlass.Int32,
    ):
        """Write one O tile, skipping rows past this sequence's last token.

        Both barriers belong to the caller's loop as much as to this store: the
        first is what makes the math warps' shared writes visible to the loads
        below, and the second is what lets the next block overwrite the buffer.
        Every thread in the block reaches both.
        """
        cute.arch.barrier()

        gO_head = gO[None, None, work_desc.o_head_idx(num_q_heads, num_v_heads)]
        tok_base = work_desc.tok_offset + blk * cutlass.Int32(self.BLK_Q)
        # One past the last token this sequence owns. The sm_90 path folds this
        # into the descriptor; here it gates the row.
        tok_end = work_desc.tok_offset + work_desc.seq_len

        # One 128-bit store a lane, replacing the eight scalar stores the
        # loop here used to emit -- its own comment already claimed a 16 B
        # access it was not making. The thread-value layout puts the vector on
        # d, which is contiguous in both the shared tile and the global tensor,
        # so the eight elements a lane moves are adjacent on both sides.
        tv_layout = cute.make_layout(
            (self.lanes_per_row, self.rows_at_a_time),
            stride=(1, self.lanes_per_row),
        )
        val_layout = cute.make_layout((self.vec, 1))
        atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            sO.element_type,
            num_bits_per_copy=self.vec * _ELEMENT_BITS,
        )
        tiled = cute.make_tiled_copy_tv(atom, tv_layout, val_layout)
        thr = tiled.get_slice(tid)

        # The global tile this block writes, as (d, token) starting at
        # tok_base. When the value dimension is split, `self.D` is the band this
        # block owns rather than the whole head, and the band starts at the
        # descriptor's row offset -- blocks of one sequence and head write
        # disjoint rows of the same tokens.
        mO = cute.domain_offset((work_desc.v_row_offset(self.D), tok_base), gO_head)
        gO_tile = cute.zipped_divide(mO, (self.D, self.BLK_Q))[
            ((None, None), (cutlass.Int32(0), cutlass.Int32(0)))
        ]

        tSrc = thr.partition_S(sO)
        tDst = thr.partition_D(gO_tile)

        thread_row = tid // cutlass.Int32(self.lanes_per_row)
        for rep in cutlass.range_constexpr(cute.size(tSrc, mode=[2])):
            row = cutlass.Int32(rep * self.rows_at_a_time) + thread_row
            # A row past this sequence's last token belongs to the next one.
            if tok_base + row < tok_end:
                # The token stride is num_o_heads * D, a multiple of the vector
                # width, so the address is on a 16 B boundary -- but it is a
                # runtime value, so the layout algebra cannot say so and the
                # atom would be rejected. Rounding up is identity here; the
                # entry point checks the premise.
                dst = cute.make_tensor(
                    tDst[None, 0, rep].iterator.align(_ALIGN_BYTES),
                    tDst[None, 0, rep].layout,
                )
                cute.copy(atom, tSrc[None, 0, rep], dst)

        cute.arch.barrier()


__all__ = ["CollectiveStoreSm80"]
