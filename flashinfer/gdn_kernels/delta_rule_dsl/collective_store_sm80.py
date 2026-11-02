"""SMEM to GMEM O store for the sm_80 delta-rule prefill kernel.

Bulk S2G copies are sm_90+, so every thread writes 16 B vectors, the tensor
bound becomes a row predicate, and block barriers replace the store pipeline.
"""

import cutlass
import cutlass.cute as cute

_ELEMENT_BITS = 16  # bf16
_ALIGN_BYTES = 16


class CollectiveStoreSm80:
    """Writes O tiles from shared memory with predicated vector stores."""

    def __init__(self, blk_q: int, d: int, num_threads: int):
        self.BLK_Q = blk_q
        self.D = d
        self.num_threads = num_threads
        # Eight bf16 is one 16 B store; there is no narrower path.
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
        tok_begin=None,
    ):
        """Write one O tile up to the sequence end; every thread must call it.

        ``tok_begin``: the sequence's first row when ``work_desc`` may start
        before it (virtual left padding); earlier rows are not written.
        """
        cute.arch.barrier()

        gO_head = gO[None, None, work_desc.o_head_idx(num_q_heads, num_v_heads)]
        tok_base = work_desc.tok_offset + blk * cutlass.Int32(self.BLK_Q)
        tok_end = work_desc.tok_offset + work_desc.seq_len

        # One 128-bit store per lane, the vector along the contiguous d.
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

        # (d, token) tile at tok_base; with V split, the block's band starts at
        # its row offset.
        mO = cute.domain_offset((work_desc.v_row_offset(self.D), tok_base), gO_head)
        gO_tile = cute.zipped_divide(mO, (self.D, self.BLK_Q))[
            ((None, None), (cutlass.Int32(0), cutlass.Int32(0)))
        ]

        tSrc = thr.partition_S(sO)
        tDst = thr.partition_D(gO_tile)

        thread_row = tid // cutlass.Int32(self.lanes_per_row)
        for rep in cutlass.range_constexpr(cute.size(tSrc, mode=[2])):
            row = cutlass.Int32(rep * self.rows_at_a_time) + thread_row
            if cutlass.const_expr(tok_begin is None):
                if tok_base + row < tok_end:
                    # The token stride is a runtime multiple of the vector width;
                    # the entry point checks it, so assert the 16 B alignment.
                    dst = cute.make_tensor(
                        tDst[None, 0, rep].iterator.align(_ALIGN_BYTES),
                        tDst[None, 0, rep].layout,
                    )
                    cute.copy(atom, tSrc[None, 0, rep], dst)
            else:
                if tok_base + row < tok_end and tok_base + row >= tok_begin:
                    dst = cute.make_tensor(
                        tDst[None, 0, rep].iterator.align(_ALIGN_BYTES),
                        tDst[None, 0, rep].layout,
                    )
                    cute.copy(atom, tSrc[None, 0, rep], dst)

        cute.arch.barrier()


__all__ = ["CollectiveStoreSm80"]
