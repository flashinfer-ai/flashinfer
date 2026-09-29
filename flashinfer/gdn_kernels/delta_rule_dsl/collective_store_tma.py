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
    ):
        o_head_idx = work_desc.o_head_idx(num_q_heads, num_v_heads)
        o_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, o_stage
        )
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
