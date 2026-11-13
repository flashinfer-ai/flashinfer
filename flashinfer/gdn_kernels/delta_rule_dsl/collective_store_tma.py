import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass._mlir.dialects.cute as _cute_ir
from cutlass.cute.nvgpu import cpasync
from cutlass.utils.tensormap_manager import TensorMapManager, TensorMapUpdateMode
from .helpers import smid, tensormap_replace_global_dim_1


class CollectiveStoreTma:
    def __init__(self, blk_q: int, d: int):
        self.BLK_Q = blk_q
        self.D = d

    @cute.jit
    def tail_tensormap_gmem_ptr(self, g_tensormaps: cute.Tensor):
        manager = TensorMapManager(TensorMapUpdateMode.GMEM, 128)
        return manager.get_tensormap_ptr(
            g_tensormaps.iterator + smid() * cutlass.Int32(128)
        )

    @cute.jit
    def tail_tensormap_generic_ptr(self, g_tensormaps: cute.Tensor):
        manager = TensorMapManager(TensorMapUpdateMode.GMEM, 128)
        return manager.get_tensormap_ptr(
            g_tensormaps.iterator + smid() * cutlass.Int32(128),
            address_space=_cute_ir.AddressSpace.generic,
        )

    @cute.jit
    def can_process(
        self,
        sO: cute.Tensor,
        work_desc,
        blk: cutlass.Int32,
        num_blocks: cutlass.Int32,
        num_seqs: cutlass.Int32,
    ):
        # 1. Intermediate full tiles always use the base descriptor.
        # 2. A full last tile also uses the base descriptor.
        # 3. The final sequence can rely on TMA OOB handling at allocation end.
        # Otherwise, the last tile of a packed non-final sequence needs a
        # temporary descriptor with the seqlen dimension shrunk to this sequence.
        can_process = work_desc.seq_idx == num_seqs - cutlass.Int32(1)
        if blk < num_blocks - cutlass.Int32(1):
            can_process = True
        if work_desc.seq_len % cutlass.Int32(self.BLK_Q) == cutlass.Int32(0):
            can_process = True
        return can_process

    @cute.jit
    def create_tensormap_for_tail(
        self,
        tma_atom_o: cute.CopyAtom,
        g_tensormaps: cute.Tensor,
        work_desc,
    ):
        tail_ptr = self.tail_tensormap_gmem_ptr(g_tensormaps)
        with cute.arch.elect_one():
            cpasync.copy_tensormap(tma_atom_o, tail_ptr)
        cute.arch.sync_warp()
        with cute.arch.elect_one():
            tensormap_replace_global_dim_1(
                tail_ptr,
                work_desc.tok_offset + work_desc.seq_len,
            )
        cute.arch.sync_warp()
        cpasync.fence_tma_desc_release()

    @cute.jit
    def partition_sd(
        self,
        sO: cute.Tensor,
        tma_atom_o: cute.CopyAtom,
        tma_tensor_o: cute.Tensor,
        work_desc,
        o_head_idx: cutlass.Int32,
        blk: cutlass.Int32,
        stage_idx: cutlass.Int32,
    ):
        mO = cute.domain_offset(
            (cutlass.Int32(0), work_desc.tok_offset + blk * cutlass.Int32(self.BLK_Q)),
            tma_tensor_o[None, None, o_head_idx],
        )
        gO = cute.zipped_divide(mO, (self.D, self.BLK_Q))[
            (
                (None, None),
                (cutlass.Int32(0), cutlass.Int32(0)),
            )
        ]
        sO_pipe = sO[None, None, stage_idx]
        return cpasync.tma_partition(
            tma_atom_o,
            0,
            cute.make_layout(1),
            cute.group_modes(sO_pipe, 0, 2),
            cute.group_modes(gO, 0, 2),
        )

    @cute.jit
    def issue_store(
        self,
        sO: cute.Tensor,
        tma_atom_o: cute.CopyAtom,
        tma_tensor_o: cute.Tensor,
        work_desc,
        o_head_idx: cutlass.Int32,
        blk: cutlass.Int32,
        stage_idx: cutlass.Int32,
    ):
        cute.arch.fence_view_async_shared()
        tOsO, tOgO = self.partition_sd(
            sO, tma_atom_o, tma_tensor_o, work_desc, o_head_idx, blk, stage_idx
        )
        cute.copy(tma_atom_o, tOsO, tOgO)
        cute.arch.cp_async_bulk_commit_group()

    @cute.jit
    def issue_tail_store(
        self,
        sO: cute.Tensor,
        tma_atom_o: cute.CopyAtom,
        tma_tensor_o: cute.Tensor,
        g_tensormaps: cute.Tensor,
        work_desc,
        o_head_idx: cutlass.Int32,
        blk: cutlass.Int32,
        stage_idx: cutlass.Int32,
    ):
        cute.arch.fence_view_async_shared()
        tOsO, tOgO = self.partition_sd(
            sO, tma_atom_o, tma_tensor_o, work_desc, o_head_idx, blk, stage_idx
        )
        tail_gmem_ptr = self.tail_tensormap_gmem_ptr(g_tensormaps)
        tail_generic_ptr = self.tail_tensormap_generic_ptr(g_tensormaps)
        cpasync.fence_tma_desc_acquire(tail_gmem_ptr)
        cute.copy(
            tma_atom_o,
            tOsO,
            tOgO,
            tma_desc_ptr=tail_generic_ptr,
        )
        cute.arch.cp_async_bulk_commit_group()

    @cute.jit
    def store_rows_simt(
        self,
        sO: cute.Tensor,
        g_o: cute.Tensor,
        o_head_idx: cutlass.Int32,
        stage_idx: cutlass.Int32,
        row_tok0,
        row_lo: cutlass.Int32,
        row_hi: cutlass.Int32,
    ):
        """Store rows [row_lo, row_hi) of one O tile with plain stores.

        Row r of the tile is token ``row_tok0 + r``. One warp; each lane moves
        8 consecutive d (one 16 B vector) of one row per step.
        """
        vec = 8
        lanes_per_row = self.D // vec
        rows_per_step = 32 // lanes_per_row
        lane = cute.arch.lane_idx()
        d0 = (lane % cutlass.Int32(lanes_per_row)) * cutlass.Int32(vec)
        row_in_step = lane // cutlass.Int32(lanes_per_row)
        sO_stage = sO[None, None, stage_idx]
        gO_head = g_o[None, None, o_head_idx]
        store_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            g_o.element_type,
            num_bits_per_copy=vec * g_o.element_type.width,
        )
        frag = cute.make_rmem_tensor(vec, g_o.element_type)
        # Runs once per padded work item: not unrolled, so the 24-register
        # store warp keeps its addresses in registers.
        for it in cutlass.range(self.BLK_Q // rows_per_step, unroll=1):
            row = it * cutlass.Int32(rows_per_step) + row_in_step
            if row >= row_lo and row < row_hi:
                for j in cutlass.range_constexpr(vec):
                    frag[j] = sO_stage[d0 + j, row]
                # O rows are 16 B aligned (the entry checks the TMA layout)
                # and d0 is a multiple of 8 elements.
                dst = cute.make_tensor(
                    cute.domain_offset((d0, row_tok0 + row), gO_head).iterator.align(
                        16
                    ),
                    cute.make_layout(vec),
                )
                cute.copy(store_atom, frag, dst)

    @cute.jit
    def step(
        self,
        sO: cute.Tensor,
        tma_atom_o: cute.CopyAtom,
        tma_tensor_o: cute.Tensor,
        g_tensormaps: cute.Tensor,
        o_pipeline,
        o_consumer_state,
        work_desc,
        o_head_idx: cutlass.Int32,
        blk: cutlass.Int32,
        num_blocks: cutlass.Int32,
        num_seqs: cutlass.Int32,
    ):
        # Full tiles can use the base descriptor. A non-final packed sequence tail
        # must shrink the seqlen dimension in a temporary descriptor so TMA OOB
        # handling stops at this sequence instead of writing into the next one.
        if blk == cutlass.Int32(0):
            if not self.can_process(
                sO, work_desc, num_blocks - cutlass.Int32(1), num_blocks, num_seqs
            ):
                self.create_tensormap_for_tail(tma_atom_o, g_tensormaps, work_desc)

        o_pipeline.consumer_wait(o_consumer_state)
        if self.can_process(sO, work_desc, blk, num_blocks, num_seqs):
            self.issue_store(
                sO,
                tma_atom_o,
                tma_tensor_o,
                work_desc,
                o_head_idx,
                blk,
                o_consumer_state.index,
            )
        else:
            self.issue_tail_store(
                sO,
                tma_atom_o,
                tma_tensor_o,
                g_tensormaps,
                work_desc,
                o_head_idx,
                blk,
                o_consumer_state.index,
            )
        cute.arch.cp_async_bulk_wait_group(0)
        o_pipeline.consumer_release(o_consumer_state)
        o_consumer_state.advance()
        return o_consumer_state

    @cute.jit
    def step_first_padded(
        self,
        sO: cute.Tensor,
        tma_atom_o: cute.CopyAtom,
        g_tensormaps: cute.Tensor,
        o_pipeline,
        o_consumer_state,
        work_desc,
        o_head_idx: cutlass.Int32,
        num_blocks: cutlass.Int32,
        num_seqs: cutlass.Int32,
        g_o: cute.Tensor,
        pad: cutlass.Int32,
    ):
        """``step`` for the first tile of a work item that starts ``pad`` > 0
        rows before its sequence (``work_desc`` spans the padded rows).

        The sequence rows of this tile are written with plain stores: a TMA
        store cannot skip the rows before the sequence, which belong to the
        previous one. Later tiles start inside the sequence and use ``step``.
        """
        if not self.can_process(
            sO, work_desc, num_blocks - cutlass.Int32(1), num_blocks, num_seqs
        ):
            self.create_tensormap_for_tail(tma_atom_o, g_tensormaps, work_desc)
        o_pipeline.consumer_wait(o_consumer_state)
        row_hi = work_desc.seq_len
        if row_hi > cutlass.Int32(self.BLK_Q):
            row_hi = cutlass.Int32(self.BLK_Q)
        self.store_rows_simt(
            sO,
            g_o,
            o_head_idx,
            o_consumer_state.index,
            work_desc.tok_offset,
            pad,
            row_hi,
        )
        o_pipeline.consumer_release(o_consumer_state)
        o_consumer_state.advance()
        return o_consumer_state

    @cute.jit
    def run(
        self,
        sO: cute.Tensor,
        tma_atom_o: cute.CopyAtom,
        tma_tensor_o: cute.Tensor,
        g_tensormaps: cute.Tensor,
        o_pipeline,
        num_blocks: cutlass.Int32,
        work_desc,
        num_seqs: cutlass.Int32,
        o_stage: int,
        num_q_heads: cutlass.Int32,
        num_v_heads: cutlass.Int32,
        g_o: cute.Tensor = None,
        pad=None,
    ):
        """Store every O tile of ``work_desc``.

        ``g_o`` (the O tensor) and ``pad`` are given only when ``work_desc``
        may start ``pad`` rows before its sequence (virtual left padding);
        the padding rows are then not written.
        """
        o_head_idx = work_desc.o_head_idx(num_q_heads, num_v_heads)
        o_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, o_stage
        )
        if cutlass.const_expr(g_o is None):
            for blk in cutlass.range(num_blocks, unroll=1):
                o_consumer_state = self.step(
                    sO,
                    tma_atom_o,
                    tma_tensor_o,
                    g_tensormaps,
                    o_pipeline,
                    o_consumer_state,
                    work_desc,
                    o_head_idx,
                    blk,
                    num_blocks,
                    num_seqs,
                )
        else:
            # Only the first tile can hold padding rows; the loop over the
            # others is the one above.
            if pad != cutlass.Int32(0):
                o_consumer_state = self.step_first_padded(
                    sO,
                    tma_atom_o,
                    g_tensormaps,
                    o_pipeline,
                    o_consumer_state,
                    work_desc,
                    o_head_idx,
                    num_blocks,
                    num_seqs,
                    g_o,
                    pad,
                )
            else:
                o_consumer_state = self.step(
                    sO,
                    tma_atom_o,
                    tma_tensor_o,
                    g_tensormaps,
                    o_pipeline,
                    o_consumer_state,
                    work_desc,
                    o_head_idx,
                    cutlass.Int32(0),
                    num_blocks,
                    num_seqs,
                )
            for blk in cutlass.range(
                cutlass.Int32(1), num_blocks, cutlass.Int32(1), unroll=1
            ):
                o_consumer_state = self.step(
                    sO,
                    tma_atom_o,
                    tma_tensor_o,
                    g_tensormaps,
                    o_pipeline,
                    o_consumer_state,
                    work_desc,
                    o_head_idx,
                    blk,
                    num_blocks,
                    num_seqs,
                )
