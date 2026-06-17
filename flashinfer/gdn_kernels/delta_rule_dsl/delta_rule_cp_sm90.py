from enum import IntEnum

import torch
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.nvgpu import warp
from cutlass.cute.nvgpu import warpgroup

from ...utils import get_device_sm_count, _get_cache_buf
from .alpha import AlphaProcessor
from .collective_inverse_hmma import CollectiveInverse
from .collective_store_tma import CollectiveStoreTma
from .custom_compile_cache import KeyedCompileMixin, cached_compile
from .helpers import (
    SM90,
    TF32,
    WarpMmaTF32Op,
    load_tensor_as_c,
    load_tensor_as_a,
    round_down,
    select_tensor_10,
)
from .varlen_helper import (
    CP_CHUNK_LEN_GRANULARITY,
    choose_cp_chunk_len_host,
    chunks_for_len,
    max_num_chunks_host,
    varlen_chunk_idx,
    varlen_chunk_valid_len,
    workspace_num_chunks_host,
)
from .delta_rule_sm90 import (
    _FullyFusedDeltaRuleSm90,
    WarpGroupRole,
    LoadStoreWarpRole,
)
from .schedule import WorkDesc


class NamedBarrier(IntEnum):
    MATH_WG0 = 4
    MATH_WG1 = 5
    MATH_SYNC = 6


class CPDeltaRuleTPrecomputeSm90(KeyedCompileMixin):
    def __init__(
        self,
        dtype: type[cutlass.Numeric] = cutlass.Float16,
        acc_dtype: type[cutlass.Numeric] = cutlass.Float32,
    ):
        self.dtype = dtype
        self.acc_dtype = acc_dtype
        self.inverse_dtype = cutlass.Float16
        self.BLK = 64
        self.D = 128
        self.manual_cache_key("dtype", "acc_dtype", "inverse_dtype", "BLK", "D")

    @cute.jit
    def load_k_tile_tma(
        self,
        tma_atom_k: cute.CopyAtom,
        tma_tensor_k: cute.Tensor,
        sK_DS: cute.Tensor,
        k_pipeline,
        k_producer_state,
        tok_offset: cutlass.Int32,
        head_idx: cutlass.Int32,
    ):
        mK = cute.domain_offset(
            (0, tok_offset),
            tma_tensor_k[None, None, head_idx],
        )
        gK = cute.zipped_divide(mK, (self.D, self.BLK))[((None, None), (0, 0))]
        tKsK, tKgK = cpasync.tma_partition(
            tma_atom_k,
            0,
            cute.make_layout(1),
            cute.group_modes(sK_DS, 0, 2),
            cute.group_modes(gK, 0, 2),
        )
        k_pipeline.producer_acquire(k_producer_state)
        cute.copy(
            tma_atom_k,
            tKgK,
            tKsK,
            tma_bar_ptr=k_pipeline.producer_get_barrier(k_producer_state),
        )
        k_pipeline.producer_commit(k_producer_state)

    @cute.jit
    def load_beta_tile(
        self,
        g_beta: cute.Tensor,
        sBeta: cute.Tensor,
        beta_pipeline,
        beta_producer_state,
        tok_offset: cutlass.Int32,
        head_idx: cutlass.Int32,
        valid_len: cutlass.Int32,
        num_heads: cutlass.Int32,
        tidx: cutlass.Int32,
    ):
        lane = tidx % 32
        beta_pipeline.producer_acquire(beta_producer_state)
        for i in cutlass.range_constexpr(2):
            row = lane + i * 32
            beta = cutlass.Float32(0.0)
            if row < valid_len:
                offset = (tok_offset + row) * num_heads + head_idx
                beta = cutlass.Float32(g_beta[offset])
            sBeta[row] = beta
        cute.arch.fence_view_async_shared()
        beta_pipeline.producer_commit(beta_producer_state)

    @cute.jit
    def store_ikk_to_smem(
        self,
        tKKrKK: cute.Tensor,
        kk_tiled_mma,
        thread_idx: cutlass.Int32,
        sKK_inv: cute.Tensor,
        tKKcMkk: cute.Tensor,
    ):
        stsm_atom = cute.make_copy_atom(
            warp.StMatrix8x8x16bOp(transpose=False, num_matrices=4), self.inverse_dtype
        )
        tiled_store = cute.make_tiled_copy_C(stsm_atom, kk_tiled_mma)
        thr_store = tiled_store.get_slice(thread_idx)
        tKKsKK = thr_store.partition_D(sKK_inv)
        tKKrKK_cv = thr_store.retile(tKKrKK)
        tKKcMkk_cv = thr_store.retile(tKKcMkk)
        tKKrIKK = cute.make_fragment_like(tKKrKK_cv, self.inverse_dtype)
        for i in cutlass.range_constexpr(cute.size(tKKrKK_cv)):
            row, col = tKKcMkk_cv[i]
            value = cutlass.Float32(0.0)
            if row > col:
                value = cutlass.Float32(tKKrKK_cv[i])
            tKKrIKK[i] = self.inverse_dtype(value)
        cute.copy(tiled_store, tKKrIKK, tKKsKK)

    @cute.jit
    def kk_epi(
        self,
        tKKrKK: cute.Tensor,
        tKKcMkk: cute.Tensor,
        sBeta: cute.Tensor,
    ):
        for i in cutlass.range_constexpr(cute.size(tKKrKK)):
            row, col = tKKcMkk[i]
            value = cutlass.Float32(0.0)
            if row > col:
                beta = cutlass.Float32(sBeta[row])
                value = cutlass.Float32(tKKrKK[i]) * beta
            tKKrKK[i] = value

    @cute.jit
    def store_t_to_gmem(
        self,
        sKK_inv: cute.Tensor,
        sBeta: cute.Tensor,
        gT_CR: cute.Tensor,
        valid_len: cutlass.Int32,
        kk_tiled_mma,
        tidx: cutlass.Int32,
    ):
        ldsm_atom = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.inverse_dtype
        )
        tiled_load = cute.make_tiled_copy_C(ldsm_atom, kk_tiled_mma)
        thr_load = tiled_load.get_slice(tidx)
        tTsT = thr_load.partition_S(sKK_inv)
        tTrInv = cute.make_rmem_tensor(
            kk_tiled_mma.partition_shape_C((self.BLK, self.BLK)), self.inverse_dtype
        )
        tTrInv_cv = thr_load.retile(tTrInv)
        cute.copy(tiled_load, tTsT, tTrInv_cv)

        store_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.dtype)
        tiled_store = cute.make_tiled_copy_C(store_atom, kk_tiled_mma)
        thr_store = tiled_store.get_slice(tidx)
        tTgT = thr_store.partition_D(gT_CR)
        cT_CR = cute.make_identity_tensor((self.BLK, self.BLK))
        tTcT = thr_store.partition_D(cT_CR)
        tTrInv_store = thr_store.retile(tTrInv)
        tTrT = cute.make_rmem_tensor(
            kk_tiled_mma.partition_shape_C((self.BLK, self.BLK)), self.dtype
        )
        tTrT_store = thr_store.retile(tTrT)

        for i in cutlass.range_constexpr(cute.size(tTrT_store)):
            col, row = tTcT[i]
            value = cutlass.Float32(0.0)
            if row < valid_len and col < valid_len:
                beta = cutlass.Float32(sBeta[row])
                value = -beta * cutlass.Float32(tTrInv_store[i])
            tTrT_store[i] = self.dtype(value)
        cute.autovec_copy(tTrT_store, tTgT)

    @cute.jit
    def __call__(
        self,
        g_k_tma: cute.Tensor,
        g_beta: cute.Tensor,
        g_t: cute.Tensor,
        cu_seqlens: cute.Tensor,
        num_k_heads: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
        total_t_blocks: cutlass.Int32,
        max_t_blocks_per_seq: cutlass.Int32,
        num_seqs: cutlass.Int32,
        stream,
    ):
        qkv_smem_layout_atom = warpgroup.make_smem_layout_atom(
            warpgroup.SmemLayoutAtomKind.K_SW128,
            self.dtype,
        )
        k_storage_layout_sd = cute.tile_to_shape(
            qkv_smem_layout_atom, (self.BLK, self.D), order=(0, 1)
        )
        k_storage_layout_ds = cute.select(k_storage_layout_sd, [1, 0])
        kk_layout_atom = cute.make_layout((8, 8), stride=(8, 1))
        kk_layout = cute.tile_to_shape(
            kk_layout_atom, (self.BLK, self.BLK), order=(0, 1)
        )
        beta_layout = cute.make_layout(self.BLK)

        tma_load_op = cpasync.CopyBulkTensorTileG2SOp()
        tma_atom_k, tma_tensor_k = cpasync.make_tiled_tma_atom(
            tma_load_op,
            g_k_tma,
            k_storage_layout_ds,
            (self.D, self.BLK),
        )
        self.tma_load_k_bytes = cute.size(k_storage_layout_ds) * (self.dtype.width // 8)

        @cute.struct
        class SharedStorage:
            k_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            beta_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            smem_k: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(k_storage_layout_sd)], 128
            ]
            smem_kk: cute.struct.Align[
                cute.struct.MemRange[self.inverse_dtype, cute.cosize(kk_layout)], 16
            ]
            smem_beta: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(beta_layout)], 16
            ]

        self.shared_storage = SharedStorage
        self.kernel(
            tma_atom_k,
            tma_tensor_k,
            g_beta,
            g_t,
            cu_seqlens,
            num_k_heads,
            num_sab_heads,
            total_t_blocks,
            num_seqs,
        ).launch(
            grid=(num_sab_heads * max_t_blocks_per_seq, num_seqs, 1),
            block=(128, 1, 1),
            max_number_threads=(128, 1, 1),
            stream=stream,
            min_blocks_per_mp=8,
        )

    @cute.kernel
    def kernel(
        self,
        tma_atom_k: cute.CopyAtom,
        tma_tensor_k: cute.Tensor,
        g_beta: cute.Tensor,
        g_t: cute.Tensor,
        cu_seqlens: cute.Tensor,
        num_k_heads: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
        total_t_blocks: cutlass.Int32,
        num_seqs: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bx, seq_idx, _ = cute.arch.block_idx()
        sab_head_idx = bx % num_sab_heads
        k_head_idx = sab_head_idx * num_k_heads // num_sab_heads
        block_idx_in_seq = bx // num_sab_heads
        seq_start = cutlass.Int32(cu_seqlens[seq_idx])
        seq_end = cutlass.Int32(cu_seqlens[seq_idx + 1])
        seq_len = seq_end - seq_start
        num_blocks = chunks_for_len(seq_len, self.BLK)
        if block_idx_in_seq < num_blocks:
            tok_offset = seq_start + block_idx_in_seq * self.BLK
            t_block_idx = varlen_chunk_idx(
                seq_idx, seq_start, block_idx_in_seq, self.BLK
            )
            valid_len = varlen_chunk_valid_len(seq_len, block_idx_in_seq, self.BLK)
            mT = cute.make_tensor(
                g_t.iterator,
                cute.make_ordered_layout(
                    (self.BLK, self.BLK, num_sab_heads, total_t_blocks),
                    order=(0, 1, 2, 3),
                ),
            )
            gT = mT[None, None, sab_head_idx, t_block_idx]

            qkv_smem_layout_atom = warpgroup.make_smem_layout_atom(
                warpgroup.SmemLayoutAtomKind.K_SW128,
                self.dtype,
            )
            k_storage_layout_sd = cute.tile_to_shape(
                qkv_smem_layout_atom, (self.BLK, self.D), order=(0, 1)
            )
            k_storage_layout_ds = cute.select(k_storage_layout_sd, [1, 0])
            kk_layout_atom = cute.make_layout((8, 8), stride=(8, 1))
            kk_layout = cute.tile_to_shape(
                kk_layout_atom, (self.BLK, self.BLK), order=(0, 1)
            )
            beta_layout = cute.make_layout(self.BLK)

            allocator = cutlass.utils.SmemAllocator()
            storage = allocator.allocate(self.shared_storage)
            sK_DS = storage.smem_k.get_tensor(
                k_storage_layout_ds.outer, swizzle=k_storage_layout_ds.inner
            )
            sK_SD = select_tensor_10(sK_DS)
            sKK_inv = storage.smem_kk.get_tensor(kk_layout)
            sBeta = storage.smem_beta.get_tensor(beta_layout)

            load_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
            tma_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 4)
            vector_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 32)
            vector_consumer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, 128
            )
            k_pipeline = pipeline.PipelineTmaAsync.create(
                barrier_storage=storage.k_mbar_ptr.data_ptr(),
                num_stages=1,
                producer_group=load_producer_group,
                consumer_group=tma_consumer_group,
                tx_count=self.tma_load_k_bytes,
                cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
            )
            beta_pipeline = pipeline.PipelineAsync.create(
                barrier_storage=storage.beta_mbar_ptr.data_ptr(),
                num_stages=1,
                producer_group=vector_producer_group,
                consumer_group=vector_consumer_group,
            )
            k_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, 1
            )
            beta_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, 1
            )
            k_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, 1
            )
            beta_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, 1
            )
            cute.arch.mbarrier_init_fence()
            cute.arch.sync_threads()

            if warp_idx == 1:
                cpasync.prefetch_descriptor(tma_atom_k)
                self.load_k_tile_tma(
                    tma_atom_k,
                    tma_tensor_k,
                    sK_DS,
                    k_pipeline,
                    k_producer_state,
                    tok_offset,
                    k_head_idx,
                )
            elif warp_idx == 2:
                self.load_beta_tile(
                    g_beta,
                    sBeta,
                    beta_pipeline,
                    beta_producer_state,
                    tok_offset,
                    sab_head_idx,
                    valid_len,
                    num_sab_heads,
                    tidx,
                )

            k_pipeline.consumer_wait(k_consumer_state)
            beta_pipeline.consumer_wait(beta_consumer_state)
            cute.arch.fence_view_async_shared()
            cute.arch.sync_threads()

            kk_tiled_mma = cute.make_tiled_mma(
                cute.make_mma_atom(
                    warpgroup.MmaF16BF16Op(
                        self.dtype,
                        self.acc_dtype,
                        (self.BLK, self.BLK, 16),
                        warpgroup.OperandSource.SMEM,
                        cute.nvgpu.OperandMajorMode.K,
                        cute.nvgpu.OperandMajorMode.K,
                    )
                ),
                cute.make_layout((1, 1, 1)),
            )
            kk_thr_mma = kk_tiled_mma.get_slice(tidx)
            tKKsA = kk_thr_mma.partition_A(sK_SD)
            tKKsB = kk_thr_mma.partition_B(sK_SD)
            tKKrA = kk_thr_mma.make_fragment_A(tKKsA)
            tKKrB = kk_thr_mma.make_fragment_B(tKKsB)
            tKKrKK = kk_thr_mma.make_fragment_C(
                kk_thr_mma.partition_shape_C((self.BLK, self.BLK))
            )
            cMkk = cute.make_identity_tensor((self.BLK, self.BLK))
            tKKcMkk = kk_thr_mma.partition_C(cMkk)

            cute.nvgpu.warpgroup.fence()
            SM90.wgmma_gemm_zero_acc(kk_tiled_mma, tKKrKK, tKKrA, tKKrB)
            cute.nvgpu.warpgroup.commit_group()
            cute.nvgpu.warpgroup.wait_group(0)
            k_pipeline.consumer_release(k_consumer_state)

            self.kk_epi(tKKrKK, tKKcMkk, sBeta)
            self.store_ikk_to_smem(tKKrKK, kk_tiled_mma, tidx, sKK_inv, tKKcMkk)
            cute.arch.barrier(barrier_id=NamedBarrier.MATH_SYNC, number_of_threads=128)
            CollectiveInverse().run(sKK_inv, NamedBarrier.MATH_SYNC)
            cute.arch.barrier(barrier_id=NamedBarrier.MATH_SYNC, number_of_threads=128)
            self.store_t_to_gmem(
                sKK_inv,
                sBeta,
                gT,
                valid_len,
                kk_tiled_mma,
                tidx,
            )
            beta_pipeline.consumer_release(beta_consumer_state)


def cp_delta_rule_t_precompute_dsl_sm90(
    k: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    total_seqlen: int,
    max_seqlen: int | None = None,
    *,
    _skip_check: bool = False,
):
    """Precompute signed, beta-folded CP T tiles on SM90.

    Flat varlen input returns `T := -(inv(IKK) beta)^T` with shape
    `(total_t_blocks, num_heads, 64, 64)`, where
    `total_t_blocks = chunk_bound(num_seqs, total_seqlen, 64)`.

    Inputs must already be contiguous. This wrapper validates that contract but
    does not materialize contiguous copies.
    """
    from cutlass.cute.runtime import from_dlpack
    import cuda.bindings.driver as cuda_driver

    if not _skip_check:
        if k.ndim != 3:
            raise RuntimeError(
                f"k must have shape (total_seqlen, num_k_heads, D), got {tuple(k.shape)}"
            )
        if beta.ndim != 2 or beta.shape[0] != k.shape[0]:
            raise RuntimeError(
                f"beta must have shape (total_seqlen, num_sab_heads), got {tuple(beta.shape)}"
            )
        if total_seqlen != k.shape[0]:
            raise RuntimeError(
                f"total_seqlen must match k.shape[0], got {total_seqlen} and {k.shape[0]}"
            )
        if cu_seqlens.dtype != torch.int64:
            raise RuntimeError(
                f"cu_seqlens must have dtype torch.int64, got {cu_seqlens.dtype}"
            )
        if not cu_seqlens.is_contiguous():
            raise RuntimeError("cu_seqlens must be contiguous")
    num_seqs = cu_seqlens.shape[0] - 1
    if max_seqlen is None:
        raise RuntimeError("max_seqlen must be provided")
    if not _skip_check and max_seqlen <= 0:
        raise RuntimeError(f"max_seqlen must be positive, got {max_seqlen}")
    _, num_k_heads, d = k.shape
    num_sab_heads = beta.shape[1]
    if not _skip_check:
        if num_sab_heads < num_k_heads or num_sab_heads % num_k_heads != 0:
            raise RuntimeError(
                "beta heads must be a positive multiple of k heads, "
                f"got beta heads={num_sab_heads} and k heads={num_k_heads}"
            )
        if k.shape[-1] != 128:
            raise RuntimeError(
                f"CPDeltaRuleTPrecomputeSm90 only supports D=128, got {k.shape[-1]}"
            )
        if k.dtype not in (torch.float16, torch.bfloat16):
            raise RuntimeError(
                f"CPDeltaRuleTPrecomputeSm90 only supports fp16/bf16 inputs, got {k.dtype}"
            )
        if beta.dtype != torch.float32:
            raise RuntimeError(f"beta must have dtype torch.float32, got {beta.dtype}")
        for name, tensor in (("k", k), ("beta", beta)):
            if not tensor.is_contiguous():
                raise RuntimeError(f"{name} must be contiguous")
    total_t_blocks = workspace_num_chunks_host(cu_seqlens, 64, total_seqlen)
    max_t_blocks_per_seq = max_num_chunks_host(max_seqlen, 64)
    t = torch.empty(
        (total_t_blocks, num_sab_heads, 64, 64), dtype=k.dtype, device=k.device
    )
    if total_t_blocks == 0:
        return t
    k_tma = k.as_strided(
        (d, total_seqlen, num_k_heads),
        (1, num_k_heads * d, d),
    )

    kernel_dtype = {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
    }[k.dtype]

    stream_val = torch.cuda.current_stream().cuda_stream
    stream = cuda_driver.CUstream(stream_val)

    enable_tvm_ffi = True
    if enable_tvm_ffi:
        from_dlpack = lambda *args, **kwargs: cute.runtime.from_dlpack(
            *args, **{**kwargs, "enable_tvm_ffi": True}
        )

    kernel = CPDeltaRuleTPrecomputeSm90(kernel_dtype)
    kernel_args = (
        from_dlpack(k_tma, assumed_align=16).mark_layout_dynamic(leading_dim=0),
        from_dlpack(beta.reshape(-1), assumed_align=16).mark_layout_dynamic(),
        from_dlpack(t.view(-1), assumed_align=128).mark_layout_dynamic(),
        from_dlpack(cu_seqlens, assumed_align=8).mark_layout_dynamic(),
        cutlass.Int32(num_k_heads),
        cutlass.Int32(num_sab_heads),
        cutlass.Int32(total_t_blocks),
        cutlass.Int32(max_t_blocks_per_seq),
        cutlass.Int32(num_seqs),
        stream,
    )
    compiled = cached_compile(
        kernel,
        *kernel_args,
        compile_options=(cute.GPUArch("sm_90a"),),
    )
    compiled(*kernel_args)
    return t


class CPDeltaRuleMNPrecomputeSm90(KeyedCompileMixin):
    class WarpGroupRole(IntEnum):
        LOAD = 0
        MATH0 = 1
        MATH1 = 2

    class LoadWarpRole(IntEnum):
        LOAD_K = 0
        LOAD_V = 1
        LOAD_T = 2
        LOAD_ALPHA = 3

    @staticmethod
    def get_register_requirements(
        max_threads_per_block: int,
        min_blocks_per_multiprocessor: int,
        num_mma_warp_groups: int,
        threads_per_warp_group: int,
    ) -> tuple[int, int]:
        reg_alloc_granularity = 8
        load_registers = 40 - 2 * reg_alloc_granularity
        total_registers = (
            round_down(
                64 * 1024 // min_blocks_per_multiprocessor,
                max_threads_per_block * reg_alloc_granularity,
            )
            // threads_per_warp_group
        )
        mma_registers = round_down(
            (total_registers - load_registers) // num_mma_warp_groups,
            reg_alloc_granularity,
        )
        return min(248, load_registers), min(248, mma_registers)

    def __init__(
        self,
        dtype: type[cutlass.Numeric] = cutlass.Float16,
        acc_dtype: type[cutlass.Numeric] = cutlass.Float32,
    ):
        self.dtype = dtype
        self.acc_dtype = acc_dtype
        self.BLK = 64
        self.D = 128
        self.k_stage = 3
        self.v_stage = 2
        self.t_stage = 2
        self.alpha_stage = 4
        self.manual_cache_key(
            "dtype",
            "acc_dtype",
            "BLK",
            "D",
            "k_stage",
            "v_stage",
            "t_stage",
            "alpha_stage",
        )

    @cute.jit
    def _math_order_init(self, wg_idx: cutlass.Int32):
        if cutlass.const_expr(wg_idx == self.WarpGroupRole.MATH1):
            cute.arch.barrier_arrive(
                barrier_id=NamedBarrier.MATH_WG0, number_of_threads=256
            )

    @cute.jit
    def _math_order_wait(self, wg_idx: cutlass.Int32):
        if cutlass.const_expr(wg_idx == self.WarpGroupRole.MATH0):
            cute.arch.barrier(barrier_id=NamedBarrier.MATH_WG0, number_of_threads=256)
        else:
            cute.arch.barrier(barrier_id=NamedBarrier.MATH_WG1, number_of_threads=256)

    @cute.jit
    def _math_order_notify(self, wg_idx: cutlass.Int32):
        if cutlass.const_expr(wg_idx == self.WarpGroupRole.MATH0):
            cute.arch.barrier_arrive(
                barrier_id=NamedBarrier.MATH_WG1, number_of_threads=256
            )
        else:
            cute.arch.barrier_arrive(
                barrier_id=NamedBarrier.MATH_WG0, number_of_threads=256
            )

    @cute.jit
    def store_acc_to_smem(
        self,
        tCrC: cute.Tensor,
        sC: cute.Tensor,
        tiled_mma,
        thread_idx: cutlass.Int32,
    ):
        stsm_atom = cute.make_copy_atom(
            warp.StMatrix8x8x16bOp(transpose=True, num_matrices=4), self.dtype
        )
        tiled_copy = cute.make_tiled_copy_C(stsm_atom, tiled_mma)
        thr_copy = tiled_copy.get_slice(thread_idx)
        tCsC = thr_copy.partition_D(sC)
        tCsC_align = cute.make_tensor(tCsC.iterator.align(16), tCsC.layout)
        tCrC_cvt = cute.make_fragment_like(tCrC, self.dtype)
        for i in cutlass.range(cute.size(tCrC), unroll_full=True):
            tCrC_cvt[i] = self.dtype(tCrC[i])
        tCrC_cvt_cv = thr_copy.retile(tCrC_cvt)
        cute.copy(tiled_copy, tCrC_cvt_cv, tCsC_align)

    @cute.jit
    def store_acc_to_gmem(
        self,
        tCrC: cute.Tensor,
        gC: cute.Tensor,
        tiled_mma,
        thread_idx: cutlass.Int32,
    ):
        copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.acc_dtype)
        tiled_copy = cute.make_tiled_copy_C(copy_atom, tiled_mma)
        thr_copy = tiled_copy.get_slice(thread_idx)
        tCgC = thr_copy.partition_D(gC)
        tCrC_cv = thr_copy.retile(tCrC)
        cute.autovec_copy(tCrC_cv, tCgC)

    @cute.jit
    def load_tensor_block_tma(
        self,
        tma_atom: cute.CopyAtom,
        tma_tensor: cute.Tensor,
        sTensor: cute.Tensor,
        tensor_pipeline,
        tensor_producer_state,
        tok_offset: cutlass.Int32,
        t_block_start: cutlass.Int32,
        head_idx: cutlass.Int32,
        blk: cutlass.Int32,
        is_transfer: bool,
    ):
        sTensor_stage = sTensor[None, None, tensor_producer_state.index]
        if cutlass.const_expr(is_transfer):
            mTensor = tma_tensor[None, None, head_idx, t_block_start + blk]
            gTensor = cute.zipped_divide(mTensor, (self.BLK, self.BLK))[
                ((None, None), (0, 0))
            ]
        else:
            block_tok = blk * self.BLK
            mTensor = cute.domain_offset(
                (0, tok_offset + block_tok),
                tma_tensor[None, None, head_idx],
            )
            gTensor = cute.zipped_divide(mTensor, (self.D, self.BLK))[
                ((None, None), (0, 0))
            ]

        tTsT, tTgT = cpasync.tma_partition(
            tma_atom,
            0,
            cute.make_layout(1),
            cute.group_modes(sTensor_stage, 0, 2),
            cute.group_modes(gTensor, 0, 2),
        )

        tensor_pipeline.producer_acquire(tensor_producer_state)
        cute.copy(
            tma_atom,
            tTgT,
            tTsT,
            tma_bar_ptr=tensor_pipeline.producer_get_barrier(tensor_producer_state),
        )
        tensor_pipeline.producer_commit(tensor_producer_state)
        tensor_producer_state.advance()
        return tensor_producer_state

    @cute.jit
    def load_alpha_block(
        self,
        g_alpha: cute.Tensor,
        sAlpha: cute.Tensor,
        tok_offset: cutlass.Int32,
        head_idx: cutlass.Int32,
        blk: cutlass.Int32,
        chunk_len: cutlass.Int32,
        num_heads: cutlass.Int32,
        alpha_stage: cutlass.Int32,
        tidx: cutlass.Int32,
    ):
        lane = tidx % 32
        sAlpha_stage = sAlpha[None, None, alpha_stage]
        for i in cutlass.range_constexpr(2):
            row = lane + i * 32
            tok = blk * self.BLK + row
            alpha = cutlass.Float32(1.0)
            if tok < chunk_len:
                alpha = cutlass.Float32(
                    g_alpha[(tok_offset + tok) * num_heads + head_idx]
                )
            sAlpha_stage[row, AlphaProcessor.CUMSUM_LOG] = alpha

    @cute.jit
    def scale_acc_by_scalar(
        self,
        tCrC: cute.Tensor,
        coeff: cutlass.Float32,
    ):
        for i in cutlass.range(cute.size(tCrC), unroll_full=True):
            tCrC[i] = cutlass.Float32(tCrC[i]) * coeff

    @cute.jit
    def compute_loop_body(
        self,
        sV_DS: cute.Tensor,
        sK_DS: cute.Tensor,
        sT: cute.Tensor,
        sX_DS: cute.Tensor,
        sAlpha: cute.Tensor,
        k_pipeline,
        k_consumer_state,
        v_pipeline,
        v_consumer_state,
        t_pipeline,
        t_consumer_state,
        alpha_pipeline,
        alpha_consumer_state,
        tTransferT: cute.Tensor,
        tStateT: cute.Tensor,
        valid_len: cutlass.Int32,
        thread_idx: cutlass.Int32,
        wg_idx: cutlass.Int32,
        is_first_block: bool,
    ):
        # ── Tiled MMAs ───────────────────────────────────────────────────────
        x_tiled_mma = cute.make_tiled_mma(
            cute.make_mma_atom(
                warpgroup.MmaF16BF16Op(
                    self.dtype,
                    self.acc_dtype,
                    (64, self.BLK, 16),
                    warpgroup.OperandSource.SMEM,
                    cute.nvgpu.OperandMajorMode.MN,
                    cute.nvgpu.OperandMajorMode.K,
                )
            ),
            cute.make_layout((2, 1, 1)),
        )
        z_tiled_mma = cute.make_tiled_mma(
            cute.make_mma_atom(
                warpgroup.MmaF16BF16Op(
                    self.dtype,
                    self.acc_dtype,
                    (64, self.BLK, 16),
                    warpgroup.OperandSource.RMEM,
                    cute.nvgpu.OperandMajorMode.K,
                    cute.nvgpu.OperandMajorMode.K,
                )
            ),
            cute.make_layout((2, 1, 1)),
        )
        y_tiled_mma = z_tiled_mma
        trans_tiled_mma = cute.make_tiled_mma(
            cute.make_mma_atom(
                warpgroup.MmaF16BF16Op(
                    self.dtype,
                    self.acc_dtype,
                    (64, self.D, 16),
                    warpgroup.OperandSource.RMEM,
                    cute.nvgpu.OperandMajorMode.K,
                    cute.nvgpu.OperandMajorMode.MN,
                )
            ),
            cute.make_layout((2, 1, 1)),
        )
        state_tiled_mma = trans_tiled_mma

        x_thr_mma = x_tiled_mma.get_slice(thread_idx)
        z_thr_mma = z_tiled_mma.get_slice(thread_idx)
        y_thr_mma = z_thr_mma
        trans_thr_mma = trans_tiled_mma.get_slice(thread_idx)
        state_thr_mma = trans_thr_mma

        cK = cute.make_identity_tensor((self.D, self.BLK))

        # ── Load current block ───────────────────────────────────────────────
        k_pipeline.consumer_wait(k_consumer_state)
        v_pipeline.consumer_wait(v_consumer_state)
        t_pipeline.consumer_wait(t_consumer_state)
        alpha_pipeline.consumer_wait(alpha_consumer_state)
        cute.arch.fence_view_async_shared()

        sK = sK_DS[None, None, k_consumer_state.index]
        sV = sV_DS[None, None, v_consumer_state.index]
        sT_blk = sT[None, None, t_consumer_state.index]
        sAlpha_blk = sAlpha[None, None, alpha_consumer_state.index]
        sK_SD = select_tensor_10(sK)

        block_coeff = cutlass.Float32(sAlpha_blk[valid_len - 1, AlphaProcessor.CUMPROD])

        # ── X = T_fused K, stored as Xt = X^T for state and transfer ────────
        tXrK = x_thr_mma.make_fragment_A(x_thr_mma.partition_A(sK))
        tXrT = x_thr_mma.make_fragment_B(x_thr_mma.partition_B(sT_blk))
        tXrX = x_thr_mma.make_fragment_C(
            x_thr_mma.partition_shape_C((self.D, self.BLK))
        )
        SM90.warpgroup_fence_operand(tXrX)
        cute.nvgpu.warpgroup.fence()
        self._math_order_wait(wg_idx)
        SM90.wgmma_gemm_zero_acc(x_tiled_mma, tXrX, tXrK, tXrT)
        cute.nvgpu.warpgroup.commit_group()
        self._math_order_notify(wg_idx)
        cute.nvgpu.warpgroup.wait_group(0)
        self.store_acc_to_smem(tXrX, sX_DS, x_tiled_mma, thread_idx)
        # ensure X is visible in shared memory for WGMMA (async proxy).
        # 2 math order barrier can guarantee it, so we don't need an extra barrier here
        cute.arch.fence_view_async_shared()

        # ── Transfer: Z = M K^T, M <- gamma_end (M + Z X) ───────────────────
        tZrK = z_thr_mma.make_fragment_B(z_thr_mma.partition_B(sK_SD))
        if cutlass.const_expr(
            is_first_block
        ):  # first iter transfer matrix is identity, skip MMA
            self._math_order_wait(wg_idx)
            tTRANSrZ = load_tensor_as_a(
                sK, trans_thr_mma, thread_idx, (self.D, self.BLK), self.dtype, False
            )
            self._math_order_notify(wg_idx)
        else:
            tZrTrans = SM90.make_acc_into_op(tTransferT, z_tiled_mma, self.dtype)
            tZrZ = z_thr_mma.make_fragment_C(
                z_thr_mma.partition_shape_C((self.D, self.BLK))
            )
            SM90.warpgroup_fence_operand(tZrTrans)
            SM90.warpgroup_fence_operand(tZrZ)
            cute.nvgpu.warpgroup.fence()
            self._math_order_wait(wg_idx)
            SM90.wgmma_gemm_zero_acc(z_tiled_mma, tZrZ, tZrTrans, tZrK)
            cute.nvgpu.warpgroup.commit_group()
            self._math_order_notify(wg_idx)
            cute.nvgpu.warpgroup.wait_group(0)
            tTRANSrZ = SM90.make_acc_into_op(tZrZ, trans_tiled_mma, self.dtype)

        tTRANSrX = trans_thr_mma.make_fragment_B(trans_thr_mma.partition_B(sX_DS))
        SM90.warpgroup_fence_operand(tTRANSrZ)
        SM90.warpgroup_fence_operand(tTransferT)
        cute.nvgpu.warpgroup.fence()
        self._math_order_wait(wg_idx)
        SM90.wgmma_gemm(trans_tiled_mma, tTransferT, tTRANSrZ, tTRANSrX, True)
        cute.nvgpu.warpgroup.commit_group()
        self._math_order_notify(wg_idx)
        cute.nvgpu.warpgroup.wait_group(0)
        self.scale_acc_by_scalar(tTransferT, block_coeff)

        # ── State: Y = gamma_end S K^T - gamma_end V^T Gamma^-1, S <- gamma_end S + Y X
        tYrY = y_thr_mma.make_fragment_C(
            y_thr_mma.partition_shape_C((self.D, self.BLK))
        )
        tYcK = y_thr_mma.partition_C(cK)
        tYrV = load_tensor_as_c(
            sV,
            y_tiled_mma,
            thread_idx,
            (self.D, self.BLK),
            self.dtype,
            False,
            self.acc_dtype,
        )
        for i in cutlass.range(cute.size(tYrY), unroll_full=True):
            _, token = tYcK[i]
            neg_end_rcp = sAlpha_blk[token, AlphaProcessor.CUMPROD_NEG_END_RCP]
            tYrV[i] = tYrV[i] * neg_end_rcp

        tYrK = tZrK
        if cutlass.const_expr(not is_first_block):
            tYrS = SM90.make_acc_into_op(tStateT, y_tiled_mma, self.dtype)
            SM90.warpgroup_fence_operand(tYrS)
            SM90.warpgroup_fence_operand(tYrY)
            cute.nvgpu.warpgroup.fence()
            self._math_order_wait(wg_idx)
            SM90.wgmma_gemm_zero_acc(y_tiled_mma, tYrY, tYrS, tYrK)
            cute.nvgpu.warpgroup.commit_group()
            self._math_order_notify(wg_idx)
            cute.nvgpu.warpgroup.wait_group(0)

            for i in cutlass.range(cute.size(tYrY), unroll_full=True):
                tYrY[i] = block_coeff * tYrY[i] + tYrV[i]
        else:
            cute.basic_copy(tYrV, tYrY)

        tSTATErY = SM90.make_acc_into_op(tYrY, state_tiled_mma, self.dtype)
        tSTATErX = state_thr_mma.make_fragment_B(state_thr_mma.partition_B(sX_DS))
        SM90.warpgroup_fence_operand(tSTATErY)
        self.scale_acc_by_scalar(tStateT, block_coeff)
        SM90.warpgroup_fence_operand(tStateT)
        cute.nvgpu.warpgroup.fence()
        self._math_order_wait(wg_idx)
        SM90.wgmma_gemm(state_tiled_mma, tStateT, tSTATErY, tSTATErX, True)
        cute.nvgpu.warpgroup.commit_group()
        self._math_order_notify(wg_idx)
        cute.nvgpu.warpgroup.wait_group(0)

        k_pipeline.consumer_release(k_consumer_state)
        k_consumer_state.advance()
        v_pipeline.consumer_release(v_consumer_state)
        v_consumer_state.advance()
        t_pipeline.consumer_release(t_consumer_state)
        t_consumer_state.advance()
        alpha_pipeline.consumer_release(alpha_consumer_state)
        alpha_consumer_state.advance()
        return (
            k_consumer_state,
            v_consumer_state,
            t_consumer_state,
            alpha_consumer_state,
        )

    @cute.jit
    def run_load_tensor_role(
        self,
        tma_atom: cute.CopyAtom,
        tma_tensor: cute.Tensor,
        sTensor: cute.Tensor,
        tensor_pipeline,
        tensor_producer_state,
        tok_offset: cutlass.Int32,
        t_block_start: cutlass.Int32,
        head_idx: cutlass.Int32,
        num_blocks: cutlass.Int32,
        is_transfer: bool,
    ):
        for blk in cutlass.range(num_blocks, unroll=1):
            tensor_producer_state = self.load_tensor_block_tma(
                tma_atom,
                tma_tensor,
                sTensor,
                tensor_pipeline,
                tensor_producer_state,
                tok_offset,
                t_block_start,
                head_idx,
                blk,
                is_transfer,
            )
        return tensor_producer_state

    @cute.jit
    def run_load_alpha_role(
        self,
        g_alpha: cute.Tensor,
        sAlpha: cute.Tensor,
        alpha_pipeline,
        alpha_producer_state,
        tok_offset: cutlass.Int32,
        head_idx: cutlass.Int32,
        chunk_len: cutlass.Int32,
        num_blocks: cutlass.Int32,
        num_heads: cutlass.Int32,
        tidx: cutlass.Int32,
    ):
        for blk in cutlass.range(num_blocks, unroll=1):
            alpha_pipeline.producer_acquire(alpha_producer_state)
            cute.arch.fence_view_async_shared()
            self.load_alpha_block(
                g_alpha,
                sAlpha,
                tok_offset,
                head_idx,
                blk,
                chunk_len,
                num_heads,
                alpha_producer_state.index,
                tidx,
            )
            AlphaProcessor().run(
                sAlpha[None, None, alpha_producer_state.index],
                cutlass.Float32(1.0),
                True,
            )
            cute.arch.fence_view_async_shared()
            alpha_pipeline.producer_commit(alpha_producer_state)
            alpha_producer_state.advance()
        return alpha_producer_state

    @cute.jit
    def run_math_role(
        self,
        sV_DS: cute.Tensor,
        sK_DS: cute.Tensor,
        sT: cute.Tensor,
        sX_DS: cute.Tensor,
        sAlpha: cute.Tensor,
        k_pipeline,
        k_consumer_state,
        v_pipeline,
        v_consumer_state,
        t_pipeline,
        t_consumer_state,
        alpha_pipeline,
        alpha_consumer_state,
        g_transfer_t: cute.Tensor,
        g_state_t: cute.Tensor,
        chunk_idx: cutlass.Int32,
        head_idx: cutlass.Int32,
        chunk_len: cutlass.Int32,
        num_blocks: cutlass.Int32,
        num_heads: cutlass.Int32,
        num_chunks: cutlass.Int32,
        wg_idx: cutlass.Int32,
        tidx: cutlass.Int32,
    ):
        acc_mma = cute.make_tiled_mma(
            cute.make_mma_atom(
                warpgroup.MmaF16BF16Op(
                    self.dtype,
                    self.acc_dtype,
                    (64, self.D, 16),
                    warpgroup.OperandSource.RMEM,
                    cute.nvgpu.OperandMajorMode.K,
                    cute.nvgpu.OperandMajorMode.MN,
                )
            ),
            cute.make_layout((2, 1, 1)),
        )
        acc_thr = acc_mma.get_slice(tidx)
        tTransferT = acc_thr.make_fragment_C(
            acc_thr.partition_shape_C((self.D, self.D))
        )
        tStateT = acc_thr.make_fragment_C(acc_thr.partition_shape_C((self.D, self.D)))
        cTransfer = cute.make_identity_tensor((self.D, self.D))
        tTransferC = acc_thr.partition_C(cTransfer)
        tStateT.fill(self.acc_dtype(0.0))
        for i in cutlass.range(cute.size(tTransferT), unroll_full=True):
            row, col = tTransferC[i]
            value = cutlass.Float32(0.0)
            if row == col:
                value = cutlass.Float32(1.0)
            tTransferT[i] = value

        valid_len = chunk_len
        if valid_len > self.BLK:
            valid_len = self.BLK

        (
            k_consumer_state,
            v_consumer_state,
            t_consumer_state,
            alpha_consumer_state,
        ) = self.compute_loop_body(
            sV_DS,
            sK_DS,
            sT,
            sX_DS,
            sAlpha,
            k_pipeline,
            k_consumer_state,
            v_pipeline,
            v_consumer_state,
            t_pipeline,
            t_consumer_state,
            alpha_pipeline,
            alpha_consumer_state,
            tTransferT,
            tStateT,
            valid_len,
            tidx,
            wg_idx,
            True,
        )

        for blk in cutlass.range(1, num_blocks, 1, unroll=1):
            block_start = blk * self.BLK
            valid_len = chunk_len - block_start
            if valid_len > self.BLK:
                valid_len = self.BLK

            (
                k_consumer_state,
                v_consumer_state,
                t_consumer_state,
                alpha_consumer_state,
            ) = self.compute_loop_body(
                sV_DS,
                sK_DS,
                sT,
                sX_DS,
                sAlpha,
                k_pipeline,
                k_consumer_state,
                v_pipeline,
                v_consumer_state,
                t_pipeline,
                t_consumer_state,
                alpha_pipeline,
                alpha_consumer_state,
                tTransferT,
                tStateT,
                valid_len,
                tidx,
                wg_idx,
                False,
            )

        out_layout = cute.make_layout(
            (self.D, self.D, num_heads, num_chunks),
            stride=(self.D, 1, self.D * self.D, self.D * self.D * num_heads),
        )
        mTransferT = cute.make_tensor(g_transfer_t.iterator, out_layout)
        mStateT = cute.make_tensor(g_state_t.iterator, out_layout)
        self.store_acc_to_gmem(
            tTransferT, mTransferT[None, None, head_idx, chunk_idx], acc_mma, tidx
        )
        self.store_acc_to_gmem(
            tStateT, mStateT[None, None, head_idx, chunk_idx], acc_mma, tidx
        )
        return (
            k_consumer_state,
            v_consumer_state,
            t_consumer_state,
            alpha_consumer_state,
        )

    @cute.jit
    def __call__(
        self,
        g_k: cute.Tensor,
        g_v: cute.Tensor,
        g_t: cute.Tensor,
        g_alpha: cute.Tensor,
        g_transfer_t: cute.Tensor,
        g_state_t: cute.Tensor,
        g_cu_seqlens: cute.Tensor,
        chunk_len: cutlass.Int32,
        num_k_heads: cutlass.Int32,
        num_v_heads: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
        total_cp_chunks: cutlass.Int32,
        max_cp_chunks_per_seq: cutlass.Int32,
        num_seqs: cutlass.Int32,
        stream,
    ):
        qkv_smem_layout_atom = warpgroup.make_smem_layout_atom(
            warpgroup.SmemLayoutAtomKind.K_SW128,
            self.dtype,
        )
        k_storage_layout_sd = cute.tile_to_shape(
            qkv_smem_layout_atom,
            (self.BLK, self.D, self.k_stage),
            order=(0, 1, 2),
        )
        k_storage_layout_ds = cute.select(k_storage_layout_sd, [1, 0, 2])
        k_tma_layout_ds = cute.slice_(k_storage_layout_ds, (None, None, 0))
        v_storage_layout_sd = cute.tile_to_shape(
            qkv_smem_layout_atom,
            (self.BLK, self.D, self.v_stage),
            order=(0, 1, 2),
        )
        v_storage_layout_ds = cute.select(v_storage_layout_sd, [1, 0, 2])
        v_tma_layout_ds = cute.slice_(v_storage_layout_ds, (None, None, 0))
        x_smem_layout_atom = cute.make_layout((8, 8), stride=(8, 1))
        x_storage_layout_sd = cute.tile_to_shape(
            x_smem_layout_atom,
            (self.BLK, self.D),
            order=(0, 1),
        )
        x_storage_layout_ds = cute.select(x_storage_layout_sd, [1, 0])
        t_storage_layout = cute.tile_to_shape(
            qkv_smem_layout_atom,
            (self.BLK, self.BLK, self.t_stage),
            order=(0, 1, 2),
        )
        t_tma_layout = cute.slice_(t_storage_layout, (None, None, 0))
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp()
        tma_atom_k, tma_tensor_k = cpasync.make_tiled_tma_atom(
            tma_load_op, g_k, k_tma_layout_ds, (self.D, self.BLK)
        )
        tma_atom_v, tma_tensor_v = cpasync.make_tiled_tma_atom(
            tma_load_op, g_v, v_tma_layout_ds, (self.D, self.BLK)
        )
        tma_atom_t, tma_tensor_t = cpasync.make_tiled_tma_atom(
            tma_load_op, g_t, t_tma_layout, (self.BLK, self.BLK)
        )
        dtype_bytes = self.dtype.width // 8
        self.tma_load_k_bytes = cute.size(k_tma_layout_ds) * dtype_bytes
        self.tma_load_v_bytes = cute.size(v_tma_layout_ds) * dtype_bytes
        self.tma_load_t_bytes = cute.size(t_tma_layout) * dtype_bytes

        @cute.struct
        class SharedStorage:
            k_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.k_stage * 2]
            v_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.v_stage * 2]
            t_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.t_stage * 2]
            alpha_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.alpha_stage * 2]
            smem_v: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(v_storage_layout_sd)], 128
            ]
            smem_k: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(k_storage_layout_sd)], 128
            ]
            smem_x: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(x_storage_layout_ds)], 128
            ]
            smem_t: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(t_storage_layout)], 16
            ]
            smem_alpha: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32,
                    self.BLK * AlphaProcessor.NUM_CHANNELS * self.alpha_stage,
                ],
                16,
            ]

        self.shared_storage = SharedStorage
        self.kernel(
            g_alpha,
            tma_atom_k,
            tma_tensor_k,
            tma_atom_v,
            tma_tensor_v,
            tma_atom_t,
            tma_tensor_t,
            g_transfer_t,
            g_state_t,
            g_cu_seqlens,
            chunk_len,
            num_k_heads,
            num_v_heads,
            num_sab_heads,
            total_cp_chunks,
            num_seqs,
        ).launch(
            grid=(num_sab_heads * max_cp_chunks_per_seq, num_seqs, 1),
            block=(384, 1, 1),
            max_number_threads=(384, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        g_alpha: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        tma_tensor_k: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        tma_tensor_v: cute.Tensor,
        tma_atom_t: cute.CopyAtom,
        tma_tensor_t: cute.Tensor,
        g_transfer_t: cute.Tensor,
        g_state_t: cute.Tensor,
        g_cu_seqlens: cute.Tensor,
        chunk_len: cutlass.Int32,
        num_k_heads: cutlass.Int32,
        num_v_heads: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
        total_cp_chunks: cutlass.Int32,
        num_seqs: cutlass.Int32,
    ):
        NUM_LOAD_WARP_GROUPS = 1
        NUM_MMA_WARP_GROUPS = 2
        THREADS_PER_WARP_GROUP = 128
        MIN_BLOCKS_PER_MP = 1
        MAX_THREADS_PER_BLOCK = (
            NUM_LOAD_WARP_GROUPS + NUM_MMA_WARP_GROUPS
        ) * THREADS_PER_WARP_GROUP
        load_registers, mma_registers = self.get_register_requirements(
            MAX_THREADS_PER_BLOCK,
            MIN_BLOCKS_PER_MP,
            NUM_MMA_WARP_GROUPS,
            THREADS_PER_WARP_GROUP,
        )

        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        warp_group_idx = cute.arch.make_warp_uniform(tidx // THREADS_PER_WARP_GROUP)
        local_tidx = tidx % THREADS_PER_WARP_GROUP
        bx, seq_idx, _ = cute.arch.block_idx()
        sab_head_idx = bx % num_sab_heads
        k_head_idx = sab_head_idx * num_k_heads // num_sab_heads
        v_head_idx = sab_head_idx * num_v_heads // num_sab_heads
        chunk_idx_in_seq = bx // num_sab_heads
        seq_start = cutlass.Int32(g_cu_seqlens[seq_idx])
        seq_end = cutlass.Int32(g_cu_seqlens[seq_idx + 1])
        seq_len = seq_end - seq_start
        num_cp_chunks = chunks_for_len(seq_len, chunk_len)
        is_valid_chunk = chunk_idx_in_seq < num_cp_chunks

        tok_offset = seq_start + chunk_idx_in_seq * chunk_len
        cp_chunk_idx = varlen_chunk_idx(seq_idx, seq_start, chunk_idx_in_seq, chunk_len)
        valid_chunk_len = varlen_chunk_valid_len(seq_len, chunk_idx_in_seq, chunk_len)
        num_blocks = (valid_chunk_len + self.BLK - 1) // self.BLK
        t_blocks_per_cp_chunk = (chunk_len + self.BLK - 1) // self.BLK
        t_block_start = varlen_chunk_idx(
            seq_idx, seq_start, chunk_idx_in_seq * t_blocks_per_cp_chunk, self.BLK
        )

        qkv_smem_layout_atom = warpgroup.make_smem_layout_atom(
            warpgroup.SmemLayoutAtomKind.K_SW128,
            self.dtype,
        )
        k_storage_layout_sd = cute.tile_to_shape(
            qkv_smem_layout_atom,
            (self.BLK, self.D, self.k_stage),
            order=(0, 1, 2),
        )
        k_storage_layout_ds = cute.select(k_storage_layout_sd, [1, 0, 2])
        v_storage_layout_sd = cute.tile_to_shape(
            qkv_smem_layout_atom,
            (self.BLK, self.D, self.v_stage),
            order=(0, 1, 2),
        )
        v_storage_layout_ds = cute.select(v_storage_layout_sd, [1, 0, 2])
        x_smem_layout_atom = cute.make_layout((8, 8), stride=(8, 1))
        x_storage_layout_sd = cute.tile_to_shape(
            x_smem_layout_atom,
            (self.BLK, self.D),
            order=(0, 1),
        )
        x_storage_layout_ds = cute.select(x_storage_layout_sd, [1, 0])
        t_storage_layout = cute.tile_to_shape(
            qkv_smem_layout_atom,
            (self.BLK, self.BLK, self.t_stage),
            order=(0, 1, 2),
        )

        alpha_layout = cute.make_layout(
            (self.BLK, AlphaProcessor.NUM_CHANNELS, self.alpha_stage)
        )

        allocator = cutlass.utils.SmemAllocator()
        storage = allocator.allocate(self.shared_storage)
        sV_DS = storage.smem_v.get_tensor(
            v_storage_layout_ds.outer, swizzle=v_storage_layout_ds.inner
        )
        sK_DS = storage.smem_k.get_tensor(
            k_storage_layout_ds.outer, swizzle=k_storage_layout_ds.inner
        )
        sX_DS = storage.smem_x.get_tensor(x_storage_layout_ds)
        sT = storage.smem_t.get_tensor(
            t_storage_layout.outer, swizzle=t_storage_layout.inner
        )
        sAlpha = storage.smem_alpha.get_tensor(alpha_layout)

        ldst_warp_role = cute.arch.make_warp_uniform(warp_idx % 4)
        if ldst_warp_role == self.LoadWarpRole.LOAD_K:
            cpasync.prefetch_descriptor(tma_atom_k)
        elif ldst_warp_role == self.LoadWarpRole.LOAD_V:
            cpasync.prefetch_descriptor(tma_atom_v)
        elif ldst_warp_role == self.LoadWarpRole.LOAD_T:
            cpasync.prefetch_descriptor(tma_atom_t)

        load_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        load_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 8)
        alpha_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 32)
        alpha_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 256)
        k_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.k_mbar_ptr.data_ptr(),
            num_stages=self.k_stage,
            producer_group=load_producer_group,
            consumer_group=load_consumer_group,
            tx_count=self.tma_load_k_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        v_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.v_mbar_ptr.data_ptr(),
            num_stages=self.v_stage,
            producer_group=load_producer_group,
            consumer_group=load_consumer_group,
            tx_count=self.tma_load_v_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        t_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.t_mbar_ptr.data_ptr(),
            num_stages=self.t_stage,
            producer_group=load_producer_group,
            consumer_group=load_consumer_group,
            tx_count=self.tma_load_t_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        alpha_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.alpha_mbar_ptr.data_ptr(),
            num_stages=self.alpha_stage,
            producer_group=alpha_producer_group,
            consumer_group=alpha_consumer_group,
        )
        k_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.k_stage
        )
        v_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.v_stage
        )
        t_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.t_stage
        )
        alpha_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.alpha_stage
        )
        k_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.k_stage
        )
        v_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.v_stage
        )
        t_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.t_stage
        )
        alpha_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.alpha_stage
        )
        cute.arch.mbarrier_init_fence()
        if warp_group_idx == self.WarpGroupRole.MATH1:
            self._math_order_init(self.WarpGroupRole.MATH1)
        cute.arch.sync_threads()

        if is_valid_chunk:
            if warp_group_idx == self.WarpGroupRole.LOAD:
                cute.arch.setmaxregister_decrease(load_registers)
                if ldst_warp_role == self.LoadWarpRole.LOAD_K:
                    k_producer_state = self.run_load_tensor_role(
                        tma_atom_k,
                        tma_tensor_k,
                        sK_DS,
                        k_pipeline,
                        k_producer_state,
                        tok_offset,
                        t_block_start,
                        k_head_idx,
                        num_blocks,
                        False,
                    )
                elif ldst_warp_role == self.LoadWarpRole.LOAD_V:
                    v_producer_state = self.run_load_tensor_role(
                        tma_atom_v,
                        tma_tensor_v,
                        sV_DS,
                        v_pipeline,
                        v_producer_state,
                        tok_offset,
                        t_block_start,
                        v_head_idx,
                        num_blocks,
                        False,
                    )
                elif ldst_warp_role == self.LoadWarpRole.LOAD_T:
                    t_producer_state = self.run_load_tensor_role(
                        tma_atom_t,
                        tma_tensor_t,
                        sT,
                        t_pipeline,
                        t_producer_state,
                        tok_offset,
                        t_block_start,
                        sab_head_idx,
                        num_blocks,
                        True,
                    )
                else:
                    alpha_producer_state = self.run_load_alpha_role(
                        g_alpha,
                        sAlpha,
                        alpha_pipeline,
                        alpha_producer_state,
                        tok_offset,
                        sab_head_idx,
                        valid_chunk_len,
                        num_blocks,
                        num_sab_heads,
                        local_tidx,
                    )
            elif warp_group_idx == self.WarpGroupRole.MATH0:
                cute.arch.setmaxregister_increase(mma_registers)
                math_tidx = tidx - THREADS_PER_WARP_GROUP
                (
                    k_consumer_state,
                    v_consumer_state,
                    t_consumer_state,
                    alpha_consumer_state,
                ) = self.run_math_role(
                    sV_DS,
                    sK_DS,
                    sT,
                    sX_DS,
                    sAlpha,
                    k_pipeline,
                    k_consumer_state,
                    v_pipeline,
                    v_consumer_state,
                    t_pipeline,
                    t_consumer_state,
                    alpha_pipeline,
                    alpha_consumer_state,
                    g_transfer_t,
                    g_state_t,
                    cp_chunk_idx,
                    sab_head_idx,
                    valid_chunk_len,
                    num_blocks,
                    num_sab_heads,
                    total_cp_chunks,
                    self.WarpGroupRole.MATH0,
                    math_tidx,
                )
            else:
                cute.arch.setmaxregister_increase(mma_registers)
                math_tidx = tidx - THREADS_PER_WARP_GROUP
                (
                    k_consumer_state,
                    v_consumer_state,
                    t_consumer_state,
                    alpha_consumer_state,
                ) = self.run_math_role(
                    sV_DS,
                    sK_DS,
                    sT,
                    sX_DS,
                    sAlpha,
                    k_pipeline,
                    k_consumer_state,
                    v_pipeline,
                    v_consumer_state,
                    t_pipeline,
                    t_consumer_state,
                    alpha_pipeline,
                    alpha_consumer_state,
                    g_transfer_t,
                    g_state_t,
                    cp_chunk_idx,
                    sab_head_idx,
                    valid_chunk_len,
                    num_blocks,
                    num_sab_heads,
                    total_cp_chunks,
                    self.WarpGroupRole.MATH1,
                    math_tidx,
                )


def cp_delta_rule_mn_precompute_dsl_sm90(
    k: torch.Tensor,
    v: torch.Tensor,
    t: torch.Tensor,
    alpha: torch.Tensor,
    cu_seqlens: torch.Tensor,
    total_seqlen: int,
    cp_chunk_len: int = 4096,
    max_seqlen: int | None = None,
    *,
    _skip_check: bool = False,
):
    """Run the SM90 CP preprocess kernel and return its native output layout.

    Flat varlen input returns transfer/state workspaces with shape
    `(total_cp_chunks, num_heads, DimV, DimK)`, where
    `total_cp_chunks = chunk_bound(num_seqs, total_seqlen, cp_chunk_len)`.

    Inputs must already be contiguous. This wrapper validates that contract but
    does not materialize contiguous copies.
    """
    from cutlass.cute.runtime import from_dlpack
    import cuda.bindings.driver as cuda_driver

    if not _skip_check:
        if k.ndim != 3:
            raise RuntimeError(
                f"k must have shape (total_seqlen, num_k_heads, D), got {tuple(k.shape)}"
            )
        if v.ndim != 3 or v.shape[0] != k.shape[0] or v.shape[2] != k.shape[2]:
            raise RuntimeError(
                f"v must have shape (total_seqlen, num_v_heads, D), got {tuple(v.shape)}"
            )
        if alpha.ndim != 2 or alpha.shape[0] != k.shape[0]:
            raise RuntimeError(
                f"alpha must have shape (total_seqlen, num_sab_heads), got {tuple(alpha.shape)}"
            )
        if total_seqlen != k.shape[0]:
            raise RuntimeError(
                f"total_seqlen must match k.shape[0], got {total_seqlen} and {k.shape[0]}"
            )
        if cp_chunk_len % 64 != 0:
            raise RuntimeError(
                f"cp_chunk_len must be a multiple of 64, got {cp_chunk_len}"
            )
        if cu_seqlens.dtype != torch.int64:
            raise RuntimeError(
                f"cu_seqlens must have dtype torch.int64, got {cu_seqlens.dtype}"
            )
        if not cu_seqlens.is_contiguous():
            raise RuntimeError("cu_seqlens must be contiguous")
    num_seqs = cu_seqlens.shape[0] - 1
    if max_seqlen is None:
        raise RuntimeError("max_seqlen must be provided")
    if not _skip_check and max_seqlen <= 0:
        raise RuntimeError(f"max_seqlen must be positive, got {max_seqlen}")
    _, num_k_heads, d = k.shape
    num_v_heads = v.shape[1]
    num_sab_heads = alpha.shape[1]
    if not _skip_check:
        if num_sab_heads < num_k_heads or num_sab_heads % num_k_heads != 0:
            raise RuntimeError(
                "alpha heads must be a positive multiple of k heads, "
                f"got alpha heads={num_sab_heads} and k heads={num_k_heads}"
            )
        if num_sab_heads < num_v_heads or num_sab_heads % num_v_heads != 0:
            raise RuntimeError(
                "alpha heads must be a positive multiple of v heads, "
                f"got alpha heads={num_sab_heads} and v heads={num_v_heads}"
            )
    total_t_blocks = workspace_num_chunks_host(cu_seqlens, 64, total_seqlen)
    if not _skip_check and t.shape != (total_t_blocks, num_sab_heads, 64, 64):
        raise RuntimeError(
            "t must have shape "
            f"{(total_t_blocks, num_sab_heads, 64, 64)}, got {tuple(t.shape)}"
        )
    total_cp_chunks = workspace_num_chunks_host(cu_seqlens, cp_chunk_len, total_seqlen)
    if not _skip_check:
        if k.shape[-1] != 128:
            raise RuntimeError(
                f"CPDeltaRuleMNPrecomputeSm90 only supports D=128, got {k.shape[-1]}"
            )
        if k.dtype not in (torch.float16, torch.bfloat16):
            raise RuntimeError(
                f"CPDeltaRuleMNPrecomputeSm90 only supports fp16/bf16 inputs, got {k.dtype}"
            )
        if t.dtype != k.dtype:
            raise RuntimeError(
                f"t dtype must match k dtype, got {t.dtype} and {k.dtype}"
            )
        for name, tensor in (("k", k), ("v", v), ("t", t), ("alpha", alpha)):
            if not tensor.is_contiguous():
                raise RuntimeError(f"{name} must be contiguous")
    max_cp_chunks_per_seq = max_num_chunks_host(max_seqlen, cp_chunk_len)
    k_tma = k.as_strided(
        (d, total_seqlen, num_k_heads),
        (1, num_k_heads * d, d),
    )
    v_tma = v.as_strided(
        (d, total_seqlen, num_v_heads),
        (1, num_v_heads * d, d),
    )
    t_tma = t.as_strided(
        (64, 64, num_sab_heads, total_t_blocks),
        (
            64,
            1,
            64 * 64,
            num_sab_heads * 64 * 64,
        ),
    )
    workspace_ctor = torch.empty if num_seqs == 1 else torch.zeros
    transfer_t = workspace_ctor(
        (total_cp_chunks, num_sab_heads, d, d), dtype=torch.float32, device=k.device
    )
    state_t = workspace_ctor(
        (total_cp_chunks, num_sab_heads, d, d), dtype=torch.float32, device=k.device
    )
    if total_cp_chunks == 0:
        return transfer_t, state_t

    kernel_dtype = {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
    }[k.dtype]

    stream_val = torch.cuda.current_stream().cuda_stream
    stream = cuda_driver.CUstream(stream_val)

    enable_tvm_ffi = True
    if enable_tvm_ffi:
        from_dlpack = lambda *args, **kwargs: cute.runtime.from_dlpack(
            *args, **{**kwargs, "enable_tvm_ffi": True}
        )

    kernel = CPDeltaRuleMNPrecomputeSm90(kernel_dtype)
    kernel_args = (
        from_dlpack(k_tma, assumed_align=16).mark_layout_dynamic(leading_dim=0),
        from_dlpack(v_tma, assumed_align=16).mark_layout_dynamic(leading_dim=0),
        from_dlpack(t_tma, assumed_align=16).mark_layout_dynamic(leading_dim=1),
        from_dlpack(alpha.view(-1), assumed_align=16).mark_layout_dynamic(),
        from_dlpack(transfer_t.view(-1), assumed_align=16).mark_layout_dynamic(),
        from_dlpack(state_t.view(-1), assumed_align=16).mark_layout_dynamic(),
        from_dlpack(cu_seqlens, assumed_align=8).mark_layout_dynamic(),
        cutlass.Int32(cp_chunk_len),
        cutlass.Int32(num_k_heads),
        cutlass.Int32(num_v_heads),
        cutlass.Int32(num_sab_heads),
        cutlass.Int32(total_cp_chunks),
        cutlass.Int32(max_cp_chunks_per_seq),
        cutlass.Int32(num_seqs),
        stream,
    )
    compiled = cached_compile(
        kernel,
        *kernel_args,
        compile_options=(cute.GPUArch("sm_90a"),),
    )
    compiled(*kernel_args)
    return transfer_t, state_t


class CPDeltaRuleFixupSm90(KeyedCompileMixin):
    class WarpGroupRole(IntEnum):
        LOAD = 0
        MATH = 1

    class LoadWarpRole(IntEnum):
        LOAD_M = 0
        LOAD_N = 1

    @staticmethod
    def get_register_requirement(
        max_threads_per_block: int,
        min_blocks_per_multiprocessor: int,
    ) -> int:
        reg_alloc_granularity = 8
        registers_per_thread = (
            round_down(
                64 * 1024 // min_blocks_per_multiprocessor,
                max_threads_per_block * reg_alloc_granularity,
            )
            // max_threads_per_block
        )
        return min(248, registers_per_thread)

    def __init__(self, needs_initial_state: bool = False):
        self.needs_initial_state = needs_initial_state
        self.D = 128
        self.rows_per_cta = 64
        self.row_ctas = 2
        self.threads_per_cta = 256
        self.num_warps = 8
        self.m_stage = 2
        self.n_stage = 2
        self.num_col_tiles = 16
        self.num_row_tiles = 4
        self.num_tiles = 64
        self.k_tiles = 16
        self.min_blocks_per_mp = 1
        self.registers_per_thread = self.get_register_requirement(
            self.threads_per_cta,
            self.min_blocks_per_mp,
        )
        self.use_3xtf32 = False
        self.manual_cache_key(
            "needs_initial_state",
            "D",
            "rows_per_cta",
            "row_ctas",
            "threads_per_cta",
            "num_warps",
            "m_stage",
            "n_stage",
            "num_col_tiles",
            "num_row_tiles",
            "num_tiles",
            "k_tiles",
            "min_blocks_per_mp",
            "registers_per_thread",
            "use_3xtf32",
        )

    @cute.jit
    def load_fixup_tma(
        self,
        tma_atom: cute.CopyAtom,
        tma_tensor: cute.Tensor,
        sTensor: cute.Tensor,
        tensor_pipeline,
        tensor_producer_state,
        chunk_start: cutlass.Int32,
        chunk_idx_in_seq: cutlass.Int32,
        head_idx: cutlass.Int32,
        row_cta_idx: cutlass.Int32,
        is_state: bool,
    ):
        chunk_idx = chunk_start + chunk_idx_in_seq
        sTensor_stage = sTensor[None, None, tensor_producer_state.index]
        if cutlass.const_expr(is_state):
            row_offset = row_cta_idx * self.rows_per_cta
            mTensor = cute.domain_offset(
                (row_offset, 0),
                tma_tensor[None, None, head_idx, chunk_idx],
            )
            gTensor = cute.zipped_divide(mTensor, (self.rows_per_cta, self.D))[
                ((None, None), (0, 0))
            ]
        else:
            gTensor = cute.zipped_divide(
                tma_tensor[None, None, head_idx, chunk_idx], (self.D, self.D)
            )[((None, None), (0, 0))]

        tTsT, tTgT = cpasync.tma_partition(
            tma_atom,
            0,
            cute.make_layout(1),
            cute.group_modes(sTensor_stage, 0, 2),
            cute.group_modes(gTensor, 0, 2),
        )
        tensor_pipeline.producer_acquire(tensor_producer_state)
        cute.copy(
            tma_atom,
            tTgT,
            tTsT,
            tma_bar_ptr=tensor_pipeline.producer_get_barrier(tensor_producer_state),
        )
        tensor_pipeline.producer_commit(tensor_producer_state)
        tensor_producer_state.advance()
        return tensor_producer_state

    @cute.jit
    def store_fixed_state(
        self,
        tiled_store_C,
        tSMrState_store: cute.Tensor,
        tSMgFixedState: cute.Tensor,
        chunk_idx: cutlass.Int32,
    ):
        tSMgFixedState_stage = tSMgFixedState[None, None, None, chunk_idx]
        tSMgFixedState_stage = cute.make_tensor(
            tSMgFixedState_stage.iterator.align(8),
            tSMgFixedState_stage.layout,
        )
        cute.copy(tiled_store_C, tSMrState_store, tSMgFixedState_stage)

    @cute.jit
    def run_fixup_loop(
        self,
        sM: cute.Tensor,
        sN: cute.Tensor,
        m_pipeline,
        m_consumer_state,
        n_pipeline,
        n_consumer_state,
        gFixedState: cute.Tensor,
        gInitialState: cute.Tensor,
        num_chunks: cutlass.Int32,
        total_cp_chunks: cutlass.Int32,
        chunk_start: cutlass.Int32,
        seq_idx: cutlass.Int32,
        head_idx: cutlass.Int32,
        row_cta_idx: cutlass.Int32,
        math_tidx: cutlass.Int32,
    ):
        warp_idx = math_tidx // 32
        lane_idx = math_tidx - warp_idx * 32
        tiled_mma = cute.make_tiled_mma(WarpMmaTF32Op((16, 8, 8)))
        copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float32)
        store_atom = cute.make_copy_atom(
            cute.nvgpu.CopyR2GOp(), cutlass.Float32, num_bits_per_copy=64
        )
        tiled_copy_C = cute.make_tiled_copy_C(copy_atom, tiled_mma)
        tiled_store_C = cute.make_tiled_copy_C(store_atom, tiled_mma)
        thr_copy_C = tiled_copy_C.get_slice(lane_idx)  # for N, local state
        thr_store_C = tiled_store_C.get_slice(lane_idx)

        tSMrS = cute.make_rmem_tensor(
            tiled_mma.partition_shape_C((16, self.D)), cutlass.Float32
        )
        tSMrM = cute.make_rmem_tensor(
            tiled_mma.partition_shape_B((self.D, self.D)), cutlass.Float32
        )
        sM_NK = select_tensor_10(sM)
        tSMrState = cute.make_rmem_tensor(
            tiled_mma.partition_shape_C((16, self.D)), cutlass.Float32
        )
        tSMrState_cv = thr_copy_C.retile(tSMrState)
        tSMrState_store = thr_store_C.retile(tSMrState)

        row_base = warp_idx * 16
        global_row_base = row_cta_idx * self.rows_per_cta + row_base
        sN_warp = cute.local_tile(sN, (16, self.D, self.n_stage), (warp_idx, 0, 0))
        gFixedState_warp = cute.local_tile(
            gFixedState[None, None, head_idx, None],
            (16, self.D, total_cp_chunks),
            (global_row_base // 16, 0, 0),
        )
        if cutlass.const_expr(self.needs_initial_state):
            gInitialState_warp = cute.local_tile(
                gInitialState[None, None, head_idx, seq_idx],
                (16, self.D),
                (global_row_base // 16, 0),
            )
        tSMsN = thr_copy_C.partition_S(sN_warp)
        tSMgFixedState = thr_store_C.partition_D(gFixedState_warp)
        if cutlass.const_expr(self.needs_initial_state):
            tSMgInitialState = thr_copy_C.partition_S(gInitialState_warp)

        start = cutlass.Int32(0)
        if cutlass.const_expr(not self.needs_initial_state):
            # NOTE: assume initial state is always zero, S_c1 = S_init @ M + N,
            # then the state after fixup for the first chunk is N automatically.

            # skip the first recurrence
            start = cutlass.Int32(1)
            # init tSMrState from N
            n_pipeline.consumer_wait(n_consumer_state)
            cute.arch.fence_view_async_shared()
            cute.autovec_copy(
                tSMsN[None, None, None, n_consumer_state.index], tSMrState_cv
            )
            n_pipeline.consumer_release(n_consumer_state)
            n_consumer_state.advance()
            # and

            self.store_fixed_state(
                tiled_store_C, tSMrState_store, tSMgFixedState, chunk_start
            )
        else:
            # do normal recurrence for the first chunk
            start = cutlass.Int32(0)

            # init tSMrState from initial_state
            cute.autovec_copy(tSMgInitialState, tSMrState_cv)

        for chunk_idx in cutlass.range(start, num_chunks, unroll=1):
            TF32.convert_tf32_c_to_kpermuted_a(tSMrState, tSMrS)
            n_pipeline.consumer_wait(n_consumer_state)
            m_pipeline.consumer_wait(m_consumer_state)
            cute.arch.fence_view_async_shared()
            cute.autovec_copy(
                tSMsN[None, None, None, n_consumer_state.index], tSMrState_cv
            )

            TF32.load_tf32_kpermuted_b(
                tSMrM, sM_NK[None, None, m_consumer_state.index], lane_idx
            )
            for iter_k in cutlass.range_constexpr(self.k_tiles):
                tSMrS_h = cute.recast_tensor(
                    tSMrS[None, None, iter_k], cutlass.TFloat32
                )
                tSMrM_h = cute.recast_tensor(
                    tSMrM[None, None, iter_k], cutlass.TFloat32
                )
                cute.gemm(tiled_mma, tSMrState, tSMrS_h, tSMrM_h, tSMrState)
                if cutlass.const_expr(self.use_3xtf32):
                    tSMrM_l = TF32.convert_fp32_to_tf32_residual(
                        tSMrM[None, None, iter_k]
                    )
                    cute.gemm(tiled_mma, tSMrState, tSMrS_h, tSMrM_l, tSMrState)
                    tSMrS_l = TF32.convert_fp32_to_tf32_residual(
                        tSMrS[None, None, iter_k]
                    )
                    tSMrM_h = cute.recast_tensor(
                        tSMrM[None, None, iter_k], cutlass.TFloat32
                    )
                    cute.gemm(tiled_mma, tSMrState, tSMrS_l, tSMrM_h, tSMrState)

            n_pipeline.consumer_release(n_consumer_state)
            n_consumer_state.advance()
            m_pipeline.consumer_release(m_consumer_state)
            m_consumer_state.advance()

            self.store_fixed_state(
                tiled_store_C, tSMrState_store, tSMgFixedState, chunk_start + chunk_idx
            )

        return m_consumer_state, n_consumer_state

    @cute.jit
    def __call__(
        self,
        g_transfer_t: cute.Tensor,
        g_local_state_t: cute.Tensor,
        g_initial_state_t: cute.Tensor,
        g_fixed_state_t: cute.Tensor,
        g_cu_seqlens: cute.Tensor,
        chunk_len: cutlass.Int32,
        total_cp_chunks: cutlass.Int32,
        num_seqs: cutlass.Int32,
        num_heads: cutlass.Int32,
        stream,
    ):
        smem_layout_atom = warpgroup.make_smem_layout_atom(
            warpgroup.SmemLayoutAtomKind.K_INTER,
            cutlass.Float32,
        )
        transfer_layout = cute.tile_to_shape(
            smem_layout_atom, (self.D, self.D, self.m_stage), order=(0, 1, 2)
        )
        state_layout = cute.tile_to_shape(
            smem_layout_atom, (self.rows_per_cta, self.D, self.n_stage), order=(0, 1, 2)
        )
        transfer_tma_layout = cute.slice_(transfer_layout, (None, None, 0))
        state_tma_layout = cute.slice_(state_layout, (None, None, 0))
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp()
        tma_atom_m, tma_tensor_m = cpasync.make_tiled_tma_atom(
            tma_load_op,
            g_transfer_t,
            transfer_tma_layout,
            (self.D, self.D),
        )
        tma_atom_n, tma_tensor_n = cpasync.make_tiled_tma_atom(
            tma_load_op,
            g_local_state_t,
            state_tma_layout,
            (self.rows_per_cta, self.D),
        )
        self.tma_load_m_bytes = cute.size(transfer_tma_layout) * 4
        self.tma_load_n_bytes = cute.size(state_tma_layout) * 4

        @cute.struct
        class SharedStorage:
            m_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.m_stage * 2]
            n_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.n_stage * 2]
            smem_m: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(transfer_layout)], 128
            ]
            smem_n: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(state_layout)], 128
            ]

        self.shared_storage = SharedStorage
        self.kernel(
            tma_atom_m,
            tma_tensor_m,
            tma_atom_n,
            tma_tensor_n,
            g_initial_state_t,
            g_fixed_state_t,
            g_cu_seqlens,
            chunk_len,
            total_cp_chunks,
            num_seqs,
            num_heads,
        ).launch(
            grid=(num_seqs * num_heads * self.row_ctas, 1, 1),
            block=(self.threads_per_cta, 1, 1),
            max_number_threads=(self.threads_per_cta, 1, 1),
            stream=stream,
            min_blocks_per_mp=self.min_blocks_per_mp,
        )

    @cute.kernel
    def kernel(
        self,
        tma_atom_m: cute.CopyAtom,
        tma_tensor_m: cute.Tensor,
        tma_atom_n: cute.CopyAtom,
        tma_tensor_n: cute.Tensor,
        g_initial_state_t: cute.Tensor,
        g_fixed_state_t: cute.Tensor,
        g_cu_seqlens: cute.Tensor,
        chunk_len: cutlass.Int32,
        total_cp_chunks: cutlass.Int32,
        num_seqs: cutlass.Int32,
        num_heads: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        warp_group_idx = cute.arch.make_warp_uniform(tidx // 128)
        local_tidx = tidx % 128
        bx, _, _ = cute.arch.block_idx()
        row_cta_idx = bx % self.row_ctas
        head_seq_idx = bx // self.row_ctas
        head_idx = head_seq_idx % num_heads
        seq_idx = head_seq_idx // num_heads
        seq_start = cutlass.Int32(g_cu_seqlens[seq_idx])
        seq_end = cutlass.Int32(g_cu_seqlens[seq_idx + 1])
        seq_len = seq_end - seq_start
        num_chunks = chunks_for_len(seq_len, chunk_len)
        chunk_start = varlen_chunk_idx(seq_idx, seq_start, 0, chunk_len)

        smem_layout_atom = warpgroup.make_smem_layout_atom(
            warpgroup.SmemLayoutAtomKind.K_INTER,
            cutlass.Float32,
        )
        transfer_layout = cute.tile_to_shape(
            smem_layout_atom, (self.D, self.D, self.m_stage), order=(0, 1, 2)
        )
        state_layout = cute.tile_to_shape(
            smem_layout_atom, (self.rows_per_cta, self.D, self.n_stage), order=(0, 1, 2)
        )
        out_layout = cute.make_layout(
            (self.D, self.D, num_heads, total_cp_chunks),
            stride=(self.D, 1, self.D * self.D, self.D * self.D * num_heads),
        )
        allocator = cutlass.utils.SmemAllocator()
        storage = allocator.allocate(self.shared_storage)
        sM = storage.smem_m.get_tensor(
            transfer_layout.outer, swizzle=transfer_layout.inner
        )
        sN = storage.smem_n.get_tensor(state_layout.outer, swizzle=state_layout.inner)
        gFixedState = cute.make_tensor(g_fixed_state_t.iterator.align(128), out_layout)
        fixed_state_layout = cute.make_layout(
            (self.D, self.D, num_heads, num_seqs),
            stride=(self.D, 1, self.D * self.D, self.D * self.D * num_heads),
        )
        if cutlass.const_expr(self.needs_initial_state):
            gInitialState = cute.make_tensor(
                g_initial_state_t.iterator, fixed_state_layout
            )
        else:
            gInitialState = gFixedState

        ldst_warp_role = cute.arch.make_warp_uniform(warp_idx % 4)
        if warp_group_idx == self.WarpGroupRole.LOAD:
            if ldst_warp_role == self.LoadWarpRole.LOAD_M:
                cpasync.prefetch_descriptor(tma_atom_m)
            elif ldst_warp_role == self.LoadWarpRole.LOAD_N:
                cpasync.prefetch_descriptor(tma_atom_n)

        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 4)
        m_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.m_mbar_ptr.data_ptr(),
            num_stages=self.m_stage,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=self.tma_load_m_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        n_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.n_mbar_ptr.data_ptr(),
            num_stages=self.n_stage,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=self.tma_load_n_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        m_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.m_stage
        )
        n_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.n_stage
        )
        m_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.m_stage
        )
        n_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.n_stage
        )
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        if warp_group_idx == self.WarpGroupRole.LOAD:
            cute.arch.setmaxregister_decrease(40)
            if ldst_warp_role == self.LoadWarpRole.LOAD_M and num_chunks > 0:
                if cutlass.const_expr(self.needs_initial_state):
                    for chunk_idx in cutlass.range(0, num_chunks, 1, unroll=1):
                        m_producer_state = self.load_fixup_tma(
                            tma_atom_m,
                            tma_tensor_m,
                            sM,
                            m_pipeline,
                            m_producer_state,
                            chunk_start,
                            chunk_idx,
                            head_idx,
                            row_cta_idx,
                            False,
                        )
                else:
                    for chunk_idx in cutlass.range(1, num_chunks, 1, unroll=1):
                        m_producer_state = self.load_fixup_tma(
                            tma_atom_m,
                            tma_tensor_m,
                            sM,
                            m_pipeline,
                            m_producer_state,
                            chunk_start,
                            chunk_idx,
                            head_idx,
                            row_cta_idx,
                            False,
                        )
            elif ldst_warp_role == self.LoadWarpRole.LOAD_N and num_chunks > 0:
                for chunk_idx in cutlass.range(0, num_chunks, 1, unroll=1):
                    n_producer_state = self.load_fixup_tma(
                        tma_atom_n,
                        tma_tensor_n,
                        sN,
                        n_pipeline,
                        n_producer_state,
                        chunk_start,
                        chunk_idx,
                        head_idx,
                        row_cta_idx,
                        True,
                    )
        else:
            cute.arch.setmaxregister_increase(self.registers_per_thread)
            if num_chunks > 0:
                math_tidx = local_tidx
                self.run_fixup_loop(
                    sM,
                    sN,
                    m_pipeline,
                    m_consumer_state,
                    n_pipeline,
                    n_consumer_state,
                    gFixedState,
                    gInitialState,
                    num_chunks,
                    total_cp_chunks,
                    chunk_start,
                    seq_idx,
                    head_idx,
                    row_cta_idx,
                    math_tidx,
                )


def cp_delta_rule_fixup_dsl_sm90(
    local_transfer: torch.Tensor,
    local_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    total_seqlen: int,
    cp_chunk_len: int = 4096,
    initial_state: torch.Tensor | None = None,
    *,
    _skip_check: bool = False,
):
    """Fix CP precompute chunk artifacts into global chunk-boundary states.

    Flat varlen workspaces use the native
    `(total_cp_chunks, num_heads, DimV, DimK)` layout produced by
    `cp_delta_rule_mn_precompute_dsl_sm90`.
    """
    from cutlass.cute.runtime import from_dlpack
    import cuda.bindings.driver as cuda_driver

    if not _skip_check:
        if local_transfer.ndim != 4:
            raise RuntimeError(
                "local_transfer must have shape (num_chunks, num_heads, DimV, DimK), "
                f"got {tuple(local_transfer.shape)}"
            )
        if local_state.shape != local_transfer.shape:
            raise RuntimeError(
                "local_state shape must match local_transfer shape, "
                f"got {tuple(local_state.shape)} and {tuple(local_transfer.shape)}"
            )
        if local_transfer.shape[-2:] != (128, 128):
            raise RuntimeError(
                f"CPDeltaRuleFixupSm90 only supports D=128, got {tuple(local_transfer.shape[-2:])}"
            )
        if local_transfer.dtype != torch.float32 or local_state.dtype != torch.float32:
            raise RuntimeError(
                "CPDeltaRuleFixupSm90 only supports float32 inputs, "
                f"got {local_transfer.dtype} and {local_state.dtype}"
            )
        if initial_state is not None:
            if initial_state.shape != (
                cu_seqlens.shape[0] - 1,
                local_transfer.shape[1],
                128,
                128,
            ):
                raise RuntimeError(
                    "initial_state must have shape "
                    f"{(cu_seqlens.shape[0] - 1, local_transfer.shape[1], 128, 128)}, got {tuple(initial_state.shape)}"
                )
            if initial_state.dtype != torch.float32:
                raise RuntimeError(
                    f"initial_state must have dtype torch.float32, got {initial_state.dtype}"
                )
        for name, tensor in (
            ("local_transfer", local_transfer),
            ("local_state", local_state),
            ("initial_state", initial_state),
        ):
            if tensor is None:
                continue
            if not tensor.is_contiguous():
                raise RuntimeError(f"{name} must be contiguous")
    total_cp_chunks, num_heads, _, _ = local_transfer.shape
    if not _skip_check:
        if cp_chunk_len <= 0:
            raise RuntimeError(f"cp_chunk_len must be positive, got {cp_chunk_len}")
        if cu_seqlens.dtype != torch.int64:
            raise RuntimeError(
                f"cu_seqlens must have dtype torch.int64, got {cu_seqlens.dtype}"
            )
        expected_chunks = workspace_num_chunks_host(
            cu_seqlens, cp_chunk_len, total_seqlen
        )
        if expected_chunks != total_cp_chunks:
            raise RuntimeError(
                "local_transfer/local_state first dim must equal "
                f"chunk_bound(num_seqs, total_seqlen, cp_chunk_len)={expected_chunks}, got {total_cp_chunks}"
            )
        if not cu_seqlens.is_contiguous():
            raise RuntimeError("cu_seqlens must be contiguous")
    num_seqs = cu_seqlens.shape[0] - 1

    fixed_state = (
        torch.empty_like(local_state)
        if num_seqs == 1
        else torch.zeros_like(local_state)
    )
    if total_cp_chunks == 0:
        return fixed_state

    stream_val = torch.cuda.current_stream().cuda_stream
    stream = cuda_driver.CUstream(stream_val)
    d = local_transfer.shape[-1]
    local_transfer_tma = local_transfer.as_strided(
        (d, d, num_heads, total_cp_chunks),
        (d, 1, d * d, num_heads * d * d),
    )
    local_state_tma = local_state.as_strided(
        (d, d, num_heads, total_cp_chunks),
        (d, 1, d * d, num_heads * d * d),
    )

    enable_tvm_ffi = True
    if enable_tvm_ffi:
        from_dlpack = lambda *args, **kwargs: cute.runtime.from_dlpack(
            *args, **{**kwargs, "enable_tvm_ffi": True}
        )

    needs_initial_state = initial_state is not None
    initial_state_cute = (
        from_dlpack(initial_state.reshape(-1), assumed_align=16).mark_layout_dynamic()
        if needs_initial_state
        else None
    )
    kernel = CPDeltaRuleFixupSm90(needs_initial_state)
    kernel_args = (
        from_dlpack(local_transfer_tma, assumed_align=128).mark_layout_dynamic(
            leading_dim=1
        ),
        from_dlpack(local_state_tma, assumed_align=128).mark_layout_dynamic(
            leading_dim=1
        ),
        initial_state_cute,
        from_dlpack(fixed_state.reshape(-1), assumed_align=128).mark_layout_dynamic(),
        from_dlpack(cu_seqlens, assumed_align=8).mark_layout_dynamic(),
        cutlass.Int32(cp_chunk_len),
        cutlass.Int32(total_cp_chunks),
        cutlass.Int32(num_seqs),
        cutlass.Int32(num_heads),
        stream,
    )
    compiled = cached_compile(
        kernel,
        *kernel_args,
        compile_options=(cute.GPUArch("sm_90a"),),
    )
    compiled(*kernel_args)
    return fixed_state


class CPDeltaRulePrefillSm90(_FullyFusedDeltaRuleSm90):
    def __init__(
        self,
        dtype: type[cutlass.Numeric] = cutlass.Float16,
        acc_dtype: type[cutlass.Numeric] = cutlass.Float32,
        needs_initial_state: bool = False,
    ):
        super().__init__(True, False, True, False, dtype, acc_dtype)
        self.needs_initial_state = needs_initial_state
        self.t_stage = 2
        self.manual_cache_key(
            "needs_alpha",
            "needs_beta",
            "needs_init_state",
            "needs_checkpointing",
            "needs_initial_state",
            "dtype",
            "acc_dtype",
            "inverse_dtype",
            "BLK_Q",
            "BLK_KV",
            "D",
            "q_stage",
            "k_stage",
            "v_stage",
            "o_stage",
            "qk_stage",
            "kk_stage",
            "alpha_beta_stage",
            "t_stage",
        )

    @cute.jit
    def load_t_tma(
        self,
        sT: cute.Tensor,
        tma_atom_t: cute.CopyAtom,
        tma_tensor_t: cute.Tensor,
        t_pipeline,
        t_producer_state,
        blk: cutlass.Int32,
        t_block_start: cutlass.Int32,
        head_idx: cutlass.Int32,
    ):
        sT_stage = sT[None, None, t_producer_state.index]
        mT = tma_tensor_t[None, None, head_idx, t_block_start + blk]
        gT = cute.zipped_divide(mT, (self.BLK_KV, self.BLK_KV))[((None, None), (0, 0))]
        tTsT, tTgT = cpasync.tma_partition(
            tma_atom_t,
            0,
            cute.make_layout(1),
            cute.group_modes(sT_stage, 0, 2),
            cute.group_modes(gT, 0, 2),
        )
        t_pipeline.producer_acquire(t_producer_state)
        cute.copy(
            tma_atom_t,
            tTgT,
            tTsT,
            tma_bar_ptr=t_pipeline.producer_get_barrier(t_producer_state),
        )
        t_pipeline.producer_commit(t_producer_state)
        t_producer_state.advance()
        return t_producer_state

    @cute.jit
    def run_load_t_role(
        self,
        sT: cute.Tensor,
        tma_atom_t: cute.CopyAtom,
        tma_tensor_t: cute.Tensor,
        t_pipeline,
        num_blocks: cutlass.Int32,
        t_block_start: cutlass.Int32,
        head_idx: cutlass.Int32,
    ):
        t_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.t_stage
        )
        for blk in cutlass.range(num_blocks, unroll=1):
            t_producer_state = self.load_t_tma(
                sT,
                tma_atom_t,
                tma_tensor_t,
                t_pipeline,
                t_producer_state,
                blk,
                t_block_start,
                head_idx,
            )

    @cute.jit
    def cp_qk_and_t_epi(
        self,
        tQKrQK: cute.Tensor,
        tQKcMqk: cute.Tensor,
        sT: cute.Tensor,
        sKK_opd: cute.Tensor,
        sAlpha: cute.Tensor,
        alpha_stage: cutlass.Int32,
        is_final_block: bool,
        B: cutlass.Int32,
        scale: cutlass.Float32,
        kk_tiled_mma,
        tKKcMkk: cute.Tensor,
        aux_tidx: cutlass.Int32,
    ):
        alpha_log = sAlpha[None, AlphaProcessor.CUMSUM_LOG, alpha_stage]
        for i in cutlass.range_constexpr(cute.size(tQKrQK)):
            s, t = tQKcMqk[i]
            alpha = cute.math.exp2(
                cutlass.Float32(alpha_log[s]) - cutlass.Float32(alpha_log[t]),
                fastmath=True,
            )
            pred = s >= t
            if cutlass.const_expr(is_final_block):
                pred = pred and (s < B or t < B)
            tQKrQK[i] = tQKrQK[i] * alpha * scale if pred else cutlass.Float32(0.0)

        stsm_atom = cute.make_copy_atom(
            warp.StMatrix8x8x16bOp(transpose=False, num_matrices=4), self.dtype
        )
        tiled_store = cute.make_tiled_copy_C(stsm_atom, kk_tiled_mma)
        thr_store = tiled_store.get_slice(aux_tidx)
        tKKsKK = thr_store.partition_D(sKK_opd)
        tKKcMkk_cv = thr_store.retile(tKKcMkk)
        tKKrT = cute.make_fragment_like(tKKsKK, self.dtype)

        for i in cutlass.range_constexpr(cute.size(tKKrT)):
            s, t = tKKcMkk_cv[i]
            value = cutlass.Float32(0.0)
            pred = s >= t
            if cutlass.const_expr(is_final_block):
                pred = pred and s < B and t < B
            if pred:
                gamma = cute.math.exp2(
                    cutlass.Float32(alpha_log[s]) - cutlass.Float32(alpha_log[t]),
                    fastmath=True,
                )
                value = -gamma * cutlass.Float32(sT[t, s])
            tKKrT[i] = self.dtype(value)
        cute.copy(tiled_store, tKKrT, tKKsKK)

    @cute.jit
    def run_aux_loop_body(
        self,
        sQK: cute.Tensor,
        sT: cute.Tensor,
        sKK_opd: cute.Tensor,
        sAlpha: cute.Tensor,
        q_pipeline,
        q_consumer_state,
        k_pipeline,
        k_consumer_state,
        t_pipeline,
        t_consumer_state,
        qk_pipeline,
        qk_producer_state,
        kk_pipeline,
        kk_producer_state,
        alpha_pipeline,
        alpha_consumer_state,
        work_desc: WorkDesc,
        scale: cutlass.Float32,
        blk: cutlass.Int32,
        is_final_block: bool,
        qk_tiled_mma,
        kk_tiled_mma,
        tQKrQ: cute.Tensor,
        tQKrK: cute.Tensor,
        tQKcMqk: cute.Tensor,
        tKKcMkk: cute.Tensor,
        aux_tidx: cutlass.Int32,
    ):
        B = self.BLK_KV
        if cutlass.const_expr(is_final_block):
            B = work_desc.seq_len - blk * self.BLK_KV

        k_pipeline.consumer_wait(k_consumer_state)
        q_pipeline.consumer_wait(q_consumer_state)
        tQKrQK = qk_tiled_mma.get_slice(aux_tidx).make_fragment_C(
            qk_tiled_mma.get_slice(aux_tidx).partition_shape_C(
                (self.BLK_Q, self.BLK_KV)
            )
        )
        cute.nvgpu.warpgroup.fence()
        SM90.wgmma_gemm_zero_acc(
            qk_tiled_mma,
            tQKrQK,
            tQKrQ[None, None, None, q_consumer_state.index],
            tQKrK[None, None, None, k_consumer_state.index],
        )
        cute.nvgpu.warpgroup.commit_group()
        cute.nvgpu.warpgroup.wait_group(0)
        k_pipeline.consumer_release(k_consumer_state)
        k_consumer_state.advance()
        q_pipeline.consumer_release(q_consumer_state)
        q_consumer_state.advance()

        alpha_pipeline.consumer_wait(alpha_consumer_state)
        t_pipeline.consumer_wait(t_consumer_state)
        cute.arch.fence_view_async_shared()

        kk_pipeline.producer_acquire(kk_producer_state)
        self.cp_qk_and_t_epi(
            tQKrQK,
            tQKcMqk,
            sT[None, None, t_consumer_state.index],
            sKK_opd[None, None, kk_producer_state.index],
            sAlpha,
            alpha_consumer_state.index,
            is_final_block,
            B,
            scale,
            kk_tiled_mma,
            tKKcMkk,
            aux_tidx,
        )
        cute.arch.fence_view_async_shared()
        kk_pipeline.producer_commit(kk_producer_state)
        kk_producer_state.advance()
        t_pipeline.consumer_release(t_consumer_state)
        t_consumer_state.advance()

        qk_pipeline.producer_acquire(qk_producer_state)
        self.qk_store(
            tQKrQK,
            sQK[None, None, qk_producer_state.index],
            qk_tiled_mma,
            aux_tidx,
        )
        cute.arch.fence_view_async_shared()
        qk_pipeline.producer_commit(qk_producer_state)
        qk_producer_state.advance()

        alpha_pipeline.consumer_release(alpha_consumer_state)
        alpha_consumer_state.advance()
        return (
            q_consumer_state,
            k_consumer_state,
            t_consumer_state,
            qk_producer_state,
            kk_producer_state,
            alpha_consumer_state,
        )

    @cute.jit
    def run_aux_math_role(
        self,
        sQ_SD: cute.Tensor,
        sK_SD: cute.Tensor,
        sQK: cute.Tensor,
        sT: cute.Tensor,
        sKK_opd: cute.Tensor,
        sAlpha: cute.Tensor,
        q_pipeline,
        k_pipeline,
        t_pipeline,
        qk_pipeline,
        kk_pipeline,
        alpha_pipeline,
        work_desc: WorkDesc,
        scale: cutlass.Float32,
        math_tidx: cutlass.Int32,
        num_blocks: cutlass.Int32,
    ):
        aux_tidx = math_tidx % 128

        qk_tiled_mma = cute.make_tiled_mma(
            cute.make_mma_atom(
                warpgroup.MmaF16BF16Op(
                    self.dtype,
                    self.acc_dtype,
                    (self.BLK_Q, self.BLK_KV, 16),
                    warpgroup.OperandSource.SMEM,
                    cute.nvgpu.OperandMajorMode.K,
                    cute.nvgpu.OperandMajorMode.K,
                )
            ),
            cute.make_layout((1, 1, 1)),
        )
        kk_tiled_mma = cute.make_tiled_mma(
            cute.make_mma_atom(
                warpgroup.MmaF16BF16Op(
                    self.dtype,
                    self.acc_dtype,
                    (self.BLK_KV, self.BLK_KV, 16),
                    warpgroup.OperandSource.SMEM,
                    cute.nvgpu.OperandMajorMode.K,
                    cute.nvgpu.OperandMajorMode.K,
                )
            ),
            cute.make_layout((1, 1, 1)),
        )

        qk_thr_mma = qk_tiled_mma.get_slice(aux_tidx)
        kk_thr_mma = kk_tiled_mma.get_slice(aux_tidx)

        tQKsQ = qk_thr_mma.partition_A(sQ_SD)
        tQKsK = qk_thr_mma.partition_B(sK_SD)
        tQKrQ = qk_thr_mma.make_fragment_A(tQKsQ)
        tQKrK = qk_thr_mma.make_fragment_B(tQKsK)

        cMqk = cute.make_identity_tensor((self.BLK_Q, self.BLK_KV))
        tQKcMqk = qk_thr_mma.partition_C(cMqk)
        tKKcMkk = kk_thr_mma.partition_C(cMqk)

        q_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.q_stage
        )
        k_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.k_stage
        )
        t_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.t_stage
        )
        qk_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.qk_stage
        )
        kk_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.kk_stage
        )
        alpha_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.alpha_beta_stage
        )

        for blk in cutlass.range(num_blocks - 1, unroll=1):
            (
                q_consumer_state,
                k_consumer_state,
                t_consumer_state,
                qk_producer_state,
                kk_producer_state,
                alpha_consumer_state,
            ) = self.run_aux_loop_body(
                sQK,
                sT,
                sKK_opd,
                sAlpha,
                q_pipeline,
                q_consumer_state,
                k_pipeline,
                k_consumer_state,
                t_pipeline,
                t_consumer_state,
                qk_pipeline,
                qk_producer_state,
                kk_pipeline,
                kk_producer_state,
                alpha_pipeline,
                alpha_consumer_state,
                work_desc,
                scale,
                blk,
                False,
                qk_tiled_mma,
                kk_tiled_mma,
                tQKrQ,
                tQKrK,
                tQKcMqk,
                tKKcMkk,
                aux_tidx,
            )

        last_blk = num_blocks - 1
        self.run_aux_loop_body(
            sQK,
            sT,
            sKK_opd,
            sAlpha,
            q_pipeline,
            q_consumer_state,
            k_pipeline,
            k_consumer_state,
            t_pipeline,
            t_consumer_state,
            qk_pipeline,
            qk_producer_state,
            kk_pipeline,
            kk_producer_state,
            alpha_pipeline,
            alpha_consumer_state,
            work_desc,
            scale,
            last_blk,
            True,
            qk_tiled_mma,
            kk_tiled_mma,
            tQKrQ,
            tQKrK,
            tQKcMqk,
            tKKcMkk,
            aux_tidx,
        )

    @cute.jit
    def run_cp_state_math_role(
        self,
        sQ_SD: cute.Tensor,
        sK_SD: cute.Tensor,
        sK_DS: cute.Tensor,
        sV_DS: cute.Tensor,
        sQK: cute.Tensor,
        sKK_inv: cute.Tensor,
        sKK_opd: cute.Tensor,
        sO: cute.Tensor,
        sAlpha: cute.Tensor,
        q_pipeline,
        k_pipeline,
        v_pipeline,
        o_pipeline,
        qk_pipeline,
        kk_pipeline,
        alpha_pipeline,
        g_state: cute.Tensor,
        g_fixed_state: cute.Tensor,
        g_initial_state: cute.Tensor,
        work_desc: WorkDesc,
        public_seq_idx: cutlass.Int32,
        fixed_state_idx: cutlass.Int32,
        load_fixed_state,
        load_initial_state,
        store_state,
        scale: cutlass.Float32,
        wg_idx: cutlass.Int32,
        math_tidx: cutlass.Int32,
        num_blocks: cutlass.Int32,
        num_q_heads: cutlass.Int32,
        num_v_heads: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
        num_chunks: cutlass.Int32,
        num_seqs: cutlass.Int32,
    ):
        self._math_order_init(wg_idx)
        q_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.q_stage
        )
        k_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.k_stage
        )
        v_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.v_stage
        )
        o_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.o_stage
        )
        qk_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.qk_stage
        )
        kk_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.kk_stage
        )
        alpha_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.alpha_beta_stage
        )

        kv_tiled_mma = cute.make_tiled_mma(
            cute.make_mma_atom(
                warpgroup.MmaF16BF16Op(
                    self.dtype,
                    self.acc_dtype,
                    (64, self.D, 16),
                    warpgroup.OperandSource.RMEM,
                    cute.nvgpu.OperandMajorMode.K,
                    cute.nvgpu.OperandMajorMode.MN,
                )
            ),
            cute.make_layout((2, 1, 1)),
        )
        kv_thr_mma = kv_tiled_mma.get_slice(math_tidx)
        tKVrKV = kv_thr_mma.make_fragment_C(
            kv_thr_mma.partition_shape_C((self.D, self.D))
        )
        tKVrKV.fill(self.acc_dtype(0.0))

        state_layout = cute.make_ordered_layout(
            (self.D, self.D, num_sab_heads, num_seqs), order=(0, 1, 2, 3)
        )
        o_head_idx = work_desc.o_head_idx(num_q_heads, num_v_heads)
        mState = cute.make_tensor(g_state.iterator, state_layout)
        gStateKV = mState[None, None, o_head_idx, public_seq_idx]
        if cutlass.const_expr(self.needs_initial_state):
            mInitialState = cute.make_tensor(g_initial_state.iterator, state_layout)
            gInitialKV = mInitialState[None, None, o_head_idx, public_seq_idx]
        fixed_state_layout = cute.make_ordered_layout(
            (self.D, self.D, num_sab_heads, num_chunks), order=(0, 1, 2, 3)
        )
        mFixedState = cute.make_tensor(g_fixed_state.iterator, fixed_state_layout)
        gFixedKV = mFixedState[None, None, o_head_idx, fixed_state_idx]
        if cutlass.const_expr(self.needs_initial_state):
            if load_initial_state:
                self.kv_load(tKVrKV, gInitialKV, kv_tiled_mma, math_tidx)
        if load_fixed_state:
            self.kv_load(tKVrKV, gFixedKV, kv_tiled_mma, math_tidx)

        first_B = work_desc.seq_len
        if first_B > self.BLK_KV:
            first_B = self.BLK_KV
        (
            q_consumer_state,
            k_consumer_state,
            v_consumer_state,
            o_producer_state,
            qk_consumer_state,
            kk_consumer_state,
            alpha_consumer_state,
        ) = self.compute_loop_body(
            sQ_SD,
            sK_SD,
            sK_DS,
            sV_DS,
            sQK,
            sKK_inv,
            sKK_opd,
            sO,
            sAlpha,
            kv_tiled_mma,
            q_pipeline,
            q_consumer_state,
            k_pipeline,
            k_consumer_state,
            v_pipeline,
            v_consumer_state,
            o_pipeline,
            o_producer_state,
            qk_pipeline,
            qk_consumer_state,
            kk_pipeline,
            kk_consumer_state,
            alpha_pipeline,
            alpha_consumer_state,
            False,
            True,
            first_B,
            tKVrKV,
            scale,
            wg_idx,
        )

        for _ in cutlass.range(1, num_blocks - 1, 1, unroll=1):
            (
                q_consumer_state,
                k_consumer_state,
                v_consumer_state,
                o_producer_state,
                qk_consumer_state,
                kk_consumer_state,
                alpha_consumer_state,
            ) = self.compute_loop_body(
                sQ_SD,
                sK_SD,
                sK_DS,
                sV_DS,
                sQK,
                sKK_inv,
                sKK_opd,
                sO,
                sAlpha,
                kv_tiled_mma,
                q_pipeline,
                q_consumer_state,
                k_pipeline,
                k_consumer_state,
                v_pipeline,
                v_consumer_state,
                o_pipeline,
                o_producer_state,
                qk_pipeline,
                qk_consumer_state,
                kk_pipeline,
                kk_consumer_state,
                alpha_pipeline,
                alpha_consumer_state,
                False,
                False,
                self.BLK_KV,
                tKVrKV,
                scale,
                wg_idx,
            )

        if num_blocks != 1:
            last_blk = num_blocks - 1
            last_B = work_desc.seq_len - last_blk * self.BLK_KV
            (
                q_consumer_state,
                k_consumer_state,
                v_consumer_state,
                o_producer_state,
                qk_consumer_state,
                kk_consumer_state,
                alpha_consumer_state,
            ) = self.compute_loop_body(
                sQ_SD,
                sK_SD,
                sK_DS,
                sV_DS,
                sQK,
                sKK_inv,
                sKK_opd,
                sO,
                sAlpha,
                kv_tiled_mma,
                q_pipeline,
                q_consumer_state,
                k_pipeline,
                k_consumer_state,
                v_pipeline,
                v_consumer_state,
                o_pipeline,
                o_producer_state,
                qk_pipeline,
                qk_consumer_state,
                kk_pipeline,
                kk_consumer_state,
                alpha_pipeline,
                alpha_consumer_state,
                False,
                True,
                last_B,
                tKVrKV,
                scale,
                wg_idx,
            )
        if store_state:
            self.kv_store(tKVrKV, gStateKV, kv_tiled_mma, math_tidx)

    @cute.jit
    def __call__(
        self,
        g_q: cute.Tensor,
        g_k: cute.Tensor,
        g_v: cute.Tensor,
        g_t: cute.Tensor,
        g_o: cute.Tensor,
        g_alpha: cute.Tensor,
        g_state: cute.Tensor,
        g_fixed_state: cute.Tensor,
        g_initial_state: cute.Tensor,
        g_tensormaps: cute.Tensor,
        g_cu_seqlens: cute.Tensor,
        scale: cutlass.Float32,
        num_q_heads: cutlass.Int32,
        num_k_heads: cutlass.Int32,
        num_v_heads: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
        chunk_len: cutlass.Int32,
        total_cp_chunks: cutlass.Int32,
        max_cp_chunks_per_seq: cutlass.Int32,
        num_seqs: cutlass.Int32,
        stream,
    ):
        qkv_smem_layout_atom = warpgroup.make_smem_layout_atom(
            warpgroup.SmemLayoutAtomKind.K_SW128,
            self.dtype,
        )
        q_storage_layout = cute.coalesce(
            cute.tile_to_shape(
                qkv_smem_layout_atom,
                (self.BLK_Q, self.D, self.q_stage),
                order=(0, 1, 2),
            ),
            target_profile=(1, 1, 1),
        )
        q_smem_layout = cute.slice_(q_storage_layout, (None, None, 0))
        k_storage_layout_sd = cute.coalesce(
            cute.tile_to_shape(
                qkv_smem_layout_atom,
                (self.BLK_KV, self.D, self.k_stage),
                order=(0, 1, 2),
            ),
            target_profile=(1, 1, 1),
        )
        k_storage_layout_ds = cute.select(k_storage_layout_sd, [1, 0, 2])
        v_storage_layout_sd = cute.coalesce(
            cute.tile_to_shape(
                qkv_smem_layout_atom,
                (self.BLK_KV, self.D, self.v_stage),
                order=(0, 1, 2),
            ),
            target_profile=(1, 1, 1),
        )
        v_storage_layout_ds = cute.select(v_storage_layout_sd, [1, 0, 2])
        k_smem_layout = cute.slice_(k_storage_layout_ds, (None, None, 0))
        v_smem_layout = cute.slice_(v_storage_layout_ds, (None, None, 0))
        o_smem_layout_atom = warpgroup.make_smem_layout_atom(
            warpgroup.SmemLayoutAtomKind.MN_SW32,
            self.dtype,
        )
        o_storage_layout = cute.tile_to_shape(
            o_smem_layout_atom,
            (self.D, self.BLK_Q, self.o_stage),
            order=(1, 0, 2),
        )
        o_smem_layout = cute.slice_(o_storage_layout, (None, None, 0))

        qk_layout_atom = cute.make_layout((8, 8), stride=(8, 1))
        t_storage_layout = cute.tile_to_shape(
            qk_layout_atom, (self.BLK_KV, self.BLK_KV, self.t_stage), order=(0, 1, 2)
        )
        t_smem_layout = cute.slice_(t_storage_layout, (None, None, 0))

        tma_load_op = cpasync.CopyBulkTensorTileG2SOp()
        tma_atom_q, tma_tensor_q = cpasync.make_tiled_tma_atom(
            tma_load_op, g_q, q_smem_layout, (self.BLK_Q, self.D)
        )
        tma_atom_k, tma_tensor_k = cpasync.make_tiled_tma_atom(
            tma_load_op, g_k, k_smem_layout, (self.D, self.BLK_KV)
        )
        tma_atom_v, tma_tensor_v = cpasync.make_tiled_tma_atom(
            tma_load_op, g_v, v_smem_layout, (self.D, self.BLK_KV)
        )
        tma_atom_t, tma_tensor_t = cpasync.make_tiled_tma_atom(
            tma_load_op, g_t, t_smem_layout, (self.BLK_KV, self.BLK_KV)
        )

        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()
        tma_atom_o, tma_tensor_o = cpasync.make_tiled_tma_atom(
            tma_store_op, g_o, o_smem_layout, (self.D, self.BLK_Q)
        )

        dtype_bytes = self.dtype.width // 8
        self.tma_load_q_bytes = cute.size(q_smem_layout) * dtype_bytes
        self.tma_load_k_bytes = cute.size(k_smem_layout) * dtype_bytes
        self.tma_load_v_bytes = cute.size(v_smem_layout) * dtype_bytes
        self.tma_load_t_bytes = cute.size(t_smem_layout) * dtype_bytes

        qk_storage_layout = cute.tile_to_shape(
            qk_layout_atom, (self.BLK_Q, self.BLK_KV, self.qk_stage), order=(0, 1, 2)
        )
        kk_storage_layout = cute.tile_to_shape(
            qk_layout_atom, (self.BLK_KV, self.BLK_KV, self.kk_stage), order=(0, 1, 2)
        )
        alpha_storage_layout = cute.make_layout(
            (self.BLK_Q, AlphaProcessor.NUM_CHANNELS, self.alpha_beta_stage)
        )

        @cute.struct
        class SharedStorage:
            q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.q_stage * 2]
            k_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.k_stage * 2]
            v_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.v_stage * 2]
            t_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.t_stage * 2]
            o_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.o_stage * 2]
            qk_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.qk_stage * 2]
            kk_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.kk_stage * 2]
            alpha_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.alpha_beta_stage * 2
            ]

            smem_q: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(q_storage_layout)], 128
            ]
            smem_k: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(k_storage_layout_sd)], 128
            ]
            smem_v: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(v_storage_layout_sd)], 128
            ]
            smem_t: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(t_storage_layout)], 16
            ]
            smem_qk: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(qk_storage_layout)], 16
            ]
            smem_kk: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(kk_storage_layout)], 16
            ]
            smem_o: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(o_storage_layout)], 128
            ]
            smem_alpha: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(alpha_storage_layout)
                ],
                16,
            ]

        self.shared_storage = SharedStorage  # type: ignore
        self.kernel(
            g_alpha,
            tma_atom_q,
            tma_tensor_q,
            tma_atom_k,
            tma_tensor_k,
            tma_atom_v,
            tma_tensor_v,
            tma_atom_t,
            tma_tensor_t,
            tma_atom_o,
            tma_tensor_o,
            g_state,
            g_fixed_state,
            g_initial_state,
            g_tensormaps,
            g_cu_seqlens,
            scale,
            num_q_heads,
            num_k_heads,
            num_v_heads,
            num_sab_heads,
            chunk_len,
            total_cp_chunks,
            num_seqs,
        ).launch(
            grid=(num_sab_heads * max_cp_chunks_per_seq, num_seqs, 1),
            block=(512, 1, 1),
            max_number_threads=(512, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        g_alpha: cute.Tensor,
        tma_atom_q: cute.CopyAtom,
        tma_tensor_q: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        tma_tensor_k: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        tma_tensor_v: cute.Tensor,
        tma_atom_t: cute.CopyAtom,
        tma_tensor_t: cute.Tensor,
        tma_atom_o: cute.CopyAtom,
        tma_tensor_o: cute.Tensor,
        g_state: cute.Tensor,
        g_fixed_state: cute.Tensor,
        g_initial_state: cute.Tensor,
        g_tensormaps: cute.Tensor,
        g_cu_seqlens: cute.Tensor,
        scale: cutlass.Float32,
        num_q_heads: cutlass.Int32,
        num_k_heads: cutlass.Int32,
        num_v_heads: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
        chunk_len: cutlass.Int32,
        total_cp_chunks: cutlass.Int32,
        num_seqs: cutlass.Int32,
    ):
        NUM_LOAD_WARP_GROUPS = 1
        NUM_STATE_MMA_WARP_GROUPS = 2
        NUM_AUX_MMA_WARP_GROUPS = 1
        THREADS_PER_WARP_GROUP = 128
        WARPS_PER_WARP_GROUP = 4
        MIN_BLOCKS_PER_MP = 1
        MAX_THREADS_PER_BLOCK = (
            NUM_LOAD_WARP_GROUPS + NUM_STATE_MMA_WARP_GROUPS + NUM_AUX_MMA_WARP_GROUPS
        ) * THREADS_PER_WARP_GROUP
        (
            load_registers,
            state_mma_registers,
            aux_mma_registers,
        ) = self.get_register_requirements(
            MAX_THREADS_PER_BLOCK,
            MIN_BLOCKS_PER_MP,
            NUM_STATE_MMA_WARP_GROUPS,
            THREADS_PER_WARP_GROUP,
        )

        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        warp_group_idx = cute.arch.make_warp_uniform(tidx // THREADS_PER_WARP_GROUP)
        ldst_warp_role = cute.arch.make_warp_uniform(warp_idx % WARPS_PER_WARP_GROUP)

        bx, seq_idx, _ = cute.arch.block_idx()
        o_head_idx = bx % num_sab_heads
        q_head_idx = o_head_idx * num_q_heads // num_sab_heads
        k_head_idx = o_head_idx * num_k_heads // num_sab_heads
        v_head_idx = o_head_idx * num_v_heads // num_sab_heads
        chunk_idx_in_seq = bx // num_sab_heads
        seq_start = cutlass.Int32(g_cu_seqlens[seq_idx])
        seq_end = cutlass.Int32(g_cu_seqlens[seq_idx + 1])
        seq_len = seq_end - seq_start
        num_cp_chunks = chunks_for_len(seq_len, chunk_len)
        is_valid_chunk = chunk_idx_in_seq < num_cp_chunks

        tok_offset = seq_start + chunk_idx_in_seq * chunk_len
        cp_chunk_idx = varlen_chunk_idx(seq_idx, seq_start, chunk_idx_in_seq, chunk_len)
        valid_chunk_len = varlen_chunk_valid_len(seq_len, chunk_idx_in_seq, chunk_len)
        num_blocks = (valid_chunk_len + self.BLK_KV - 1) // self.BLK_KV
        t_blocks_per_cp_chunk = (chunk_len + self.BLK_KV - 1) // self.BLK_KV
        t_block_start = varlen_chunk_idx(
            seq_idx, seq_start, chunk_idx_in_seq * t_blocks_per_cp_chunk, self.BLK_KV
        )
        work_desc = WorkDesc(
            seq_idx=cp_chunk_idx,
            private_q_head_idx=q_head_idx,
            private_v_head_idx=v_head_idx,
            tok_offset=tok_offset,
            seq_len=valid_chunk_len,
            tile_idx=0,
        )
        load_fixed_state = chunk_idx_in_seq != 0
        load_initial_state = chunk_idx_in_seq == 0
        fixed_state_idx = cp_chunk_idx
        if load_fixed_state:
            fixed_state_idx = cp_chunk_idx - 1
        store_state = chunk_idx_in_seq == num_cp_chunks - 1

        if warp_idx == LoadStoreWarpRole.LOAD_QKV:
            cpasync.prefetch_descriptor(tma_atom_q)
            cpasync.prefetch_descriptor(tma_atom_k)
            cpasync.prefetch_descriptor(tma_atom_v)
            cpasync.prefetch_descriptor(tma_atom_o)
        elif warp_idx == LoadStoreWarpRole.LOAD_BETA:
            cpasync.prefetch_descriptor(tma_atom_t)

        math_tidx = tidx - THREADS_PER_WARP_GROUP
        wg_idx = math_tidx // THREADS_PER_WARP_GROUP

        allocator = cutlass.utils.SmemAllocator()
        storage = allocator.allocate(self.shared_storage)

        qkv_smem_layout_atom = warpgroup.make_smem_layout_atom(
            warpgroup.SmemLayoutAtomKind.K_SW128,
            self.dtype,
        )
        q_layout_sd = cute.coalesce(
            cute.tile_to_shape(
                qkv_smem_layout_atom,
                (self.BLK_Q, self.D, self.q_stage),
                order=(0, 1, 2),
            ),
            target_profile=(1, 1, 1),
        )
        sQ_SD = storage.smem_q.get_tensor(q_layout_sd.outer, swizzle=q_layout_sd.inner)

        k_layout_sd = cute.coalesce(
            cute.tile_to_shape(
                qkv_smem_layout_atom,
                (self.BLK_KV, self.D, self.k_stage),
                order=(0, 1, 2),
            ),
            target_profile=(1, 1, 1),
        )
        k_layout_ds = cute.select(k_layout_sd, [1, 0, 2])
        sK_SD = storage.smem_k.get_tensor(k_layout_sd.outer, swizzle=k_layout_sd.inner)
        sK_DS = storage.smem_k.get_tensor(k_layout_ds.outer, swizzle=k_layout_ds.inner)

        v_layout_sd = cute.coalesce(
            cute.tile_to_shape(
                qkv_smem_layout_atom,
                (self.BLK_KV, self.D, self.v_stage),
                order=(0, 1, 2),
            ),
            target_profile=(1, 1, 1),
        )
        v_layout_ds = cute.select(v_layout_sd, [1, 0, 2])
        sV_DS = storage.smem_v.get_tensor(v_layout_ds.outer, swizzle=v_layout_ds.inner)

        qk_layout_atom = cute.make_layout((8, 8), stride=(8, 1))
        qk_layout = cute.tile_to_shape(
            qk_layout_atom, (self.BLK_Q, self.BLK_KV, self.qk_stage), order=(0, 1, 2)
        )
        kk_layout = cute.tile_to_shape(
            qk_layout_atom, (self.BLK_KV, self.BLK_KV, self.kk_stage), order=(0, 1, 2)
        )
        t_layout = cute.tile_to_shape(
            qk_layout_atom, (self.BLK_KV, self.BLK_KV, self.t_stage), order=(0, 1, 2)
        )
        sQK = storage.smem_qk.get_tensor(qk_layout)
        sKK_opd = storage.smem_kk.get_tensor(kk_layout)
        sKK_inv = sKK_opd
        sT = storage.smem_t.get_tensor(t_layout)

        o_smem_layout_atom = warpgroup.make_smem_layout_atom(
            warpgroup.SmemLayoutAtomKind.MN_SW32,
            self.dtype,
        )
        o_layout = cute.tile_to_shape(
            o_smem_layout_atom,
            (self.D, self.BLK_Q, self.o_stage),
            order=(1, 0, 2),
        )
        sO = storage.smem_o.get_tensor(o_layout.outer, swizzle=o_layout.inner)
        alpha_layout = cute.make_layout(
            (self.BLK_Q, AlphaProcessor.NUM_CHANNELS, self.alpha_beta_stage)
        )
        sAlpha = storage.smem_alpha.get_tensor(alpha_layout)

        load_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        qk_load_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            (NUM_STATE_MMA_WARP_GROUPS + NUM_AUX_MMA_WARP_GROUPS)
            * WARPS_PER_WARP_GROUP,
        )
        v_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            NUM_STATE_MMA_WARP_GROUPS * WARPS_PER_WARP_GROUP,
        )
        t_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            NUM_AUX_MMA_WARP_GROUPS * WARPS_PER_WARP_GROUP,
        )
        vector_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 32)
        alpha_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            (NUM_AUX_MMA_WARP_GROUPS + NUM_STATE_MMA_WARP_GROUPS)
            * THREADS_PER_WARP_GROUP,
        )
        o_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, NUM_STATE_MMA_WARP_GROUPS * THREADS_PER_WARP_GROUP
        )
        o_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 32)
        q_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.q_mbar_ptr.data_ptr(),
            num_stages=self.q_stage,
            producer_group=load_producer_group,
            consumer_group=qk_load_consumer_group,
            tx_count=self.tma_load_q_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        k_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.k_mbar_ptr.data_ptr(),
            num_stages=self.k_stage,
            producer_group=load_producer_group,
            consumer_group=qk_load_consumer_group,
            tx_count=self.tma_load_k_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        v_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.v_mbar_ptr.data_ptr(),
            num_stages=self.v_stage,
            producer_group=load_producer_group,
            consumer_group=v_consumer_group,
            tx_count=self.tma_load_v_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        t_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.t_mbar_ptr.data_ptr(),
            num_stages=self.t_stage,
            producer_group=load_producer_group,
            consumer_group=t_consumer_group,
            tx_count=self.tma_load_t_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        o_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.o_mbar_ptr.data_ptr(),
            num_stages=self.o_stage,
            producer_group=o_producer_group,
            consumer_group=o_consumer_group,
        )
        qk_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.qk_mbar_ptr.data_ptr(),
            num_stages=self.qk_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                NUM_AUX_MMA_WARP_GROUPS * THREADS_PER_WARP_GROUP,
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                NUM_STATE_MMA_WARP_GROUPS * THREADS_PER_WARP_GROUP,
            ),
        )
        kk_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.kk_mbar_ptr.data_ptr(),
            num_stages=self.kk_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                NUM_AUX_MMA_WARP_GROUPS * THREADS_PER_WARP_GROUP,
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                NUM_STATE_MMA_WARP_GROUPS * THREADS_PER_WARP_GROUP,
            ),
        )
        alpha_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.alpha_mbar_ptr.data_ptr(),
            num_stages=self.alpha_beta_stage,
            producer_group=vector_producer_group,
            consumer_group=alpha_consumer_group,
        )
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        if is_valid_chunk:
            if warp_group_idx == WarpGroupRole.LDST:
                cute.arch.setmaxregister_decrease(load_registers)
                if ldst_warp_role == LoadStoreWarpRole.LOAD_QKV:
                    self.run_load_qkv_role(
                        sQ_SD,
                        sK_DS,
                        sV_DS,
                        tma_atom_q,
                        tma_tensor_q,
                        tma_atom_k,
                        tma_tensor_k,
                        tma_atom_v,
                        tma_tensor_v,
                        q_pipeline,
                        k_pipeline,
                        v_pipeline,
                        num_blocks,
                        work_desc.tok_offset,
                        q_head_idx,
                        k_head_idx,
                        v_head_idx,
                    )
                elif ldst_warp_role == LoadStoreWarpRole.STORE_O:
                    CollectiveStoreTma(self.BLK_Q, self.D).run(
                        sO,
                        tma_atom_o,
                        tma_tensor_o,
                        g_tensormaps,
                        o_pipeline,
                        num_blocks,
                        work_desc,
                        total_cp_chunks,
                        self.o_stage,
                        num_q_heads,
                        num_v_heads,
                    )
                elif ldst_warp_role == LoadStoreWarpRole.LOAD_BETA:
                    self.run_load_t_role(
                        sT,
                        tma_atom_t,
                        tma_tensor_t,
                        t_pipeline,
                        num_blocks,
                        t_block_start,
                        o_head_idx,
                    )
                elif ldst_warp_role == LoadStoreWarpRole.LOAD_ALPHA:
                    self.run_load_alpha_role(
                        sAlpha,
                        g_alpha,
                        alpha_pipeline,
                        scale,
                        num_blocks,
                        work_desc.tok_offset,
                        work_desc.tok_offset + work_desc.seq_len,
                        o_head_idx,
                        num_sab_heads,
                    )
            else:
                if warp_group_idx == WarpGroupRole.MATH_AUX:
                    cute.arch.setmaxregister_decrease(aux_mma_registers)
                    self.run_aux_math_role(
                        sQ_SD,
                        sK_SD,
                        sQK,
                        sT,
                        sKK_opd,
                        sAlpha,
                        q_pipeline,
                        k_pipeline,
                        t_pipeline,
                        qk_pipeline,
                        kk_pipeline,
                        alpha_pipeline,
                        work_desc,
                        scale,
                        math_tidx,
                        num_blocks,
                    )
                else:
                    cute.arch.setmaxregister_increase(state_mma_registers)
                    self.run_cp_state_math_role(
                        sQ_SD,
                        sK_SD,
                        sK_DS,
                        sV_DS,
                        sQK,
                        sKK_inv,
                        sKK_opd,
                        sO,
                        sAlpha,
                        q_pipeline,
                        k_pipeline,
                        v_pipeline,
                        o_pipeline,
                        qk_pipeline,
                        kk_pipeline,
                        alpha_pipeline,
                        g_state,
                        g_fixed_state,
                        g_initial_state,
                        work_desc,
                        seq_idx,
                        fixed_state_idx,
                        load_fixed_state,
                        load_initial_state,
                        store_state,
                        scale,
                        wg_idx,
                        math_tidx,
                        num_blocks,
                        num_q_heads,
                        num_v_heads,
                        num_sab_heads,
                        total_cp_chunks,
                        num_seqs,
                    )


def cp_delta_rule_prefill_dsl_sm90(
    o: torch.Tensor,
    state: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    t: torch.Tensor,
    fixed_state: torch.Tensor,
    alpha: torch.Tensor,
    scale: float,
    cu_seqlens: torch.Tensor,
    total_seqlen: int,
    cp_chunk_len: int = 4096,
    max_seqlen: int | None = None,
    initial_state: torch.Tensor | None = None,
    *,
    _skip_check: bool = False,
):
    """Run CP main prefill with precomputed T and fixed-up chunk states.

    Flat varlen input consumes flat Q/K/V/O/alpha tensors and varlen T/fixed-state
    workspaces. `state` is the public per-sequence final state in native
    `(DimV, DimK)` layout.
    """
    from cutlass.cute.runtime import from_dlpack
    import cuda.bindings.driver as cuda_driver

    if not _skip_check:
        if q.ndim != 3:
            raise RuntimeError(
                f"q must have shape (total_seqlen, num_q_heads, D), got {tuple(q.shape)}"
            )
        if k.ndim != 3 or v.ndim != 3 or o.ndim != 3:
            raise RuntimeError(
                "k, v, and o must have shape (total_seqlen, num_heads, D), "
                f"got k={tuple(k.shape)}, v={tuple(v.shape)}, o={tuple(o.shape)}"
            )
        if (
            k.shape[0] != q.shape[0]
            or v.shape[0] != q.shape[0]
            or o.shape[0] != q.shape[0]
        ):
            raise RuntimeError("q, k, v, and o must have the same total_seqlen")
        if (
            k.shape[2] != q.shape[2]
            or v.shape[2] != q.shape[2]
            or o.shape[2] != q.shape[2]
        ):
            raise RuntimeError("q, k, v, and o must have the same D")
        if alpha.ndim != 2 or alpha.shape[0] != q.shape[0]:
            raise RuntimeError(
                f"alpha must have shape (total_seqlen, num_sab_heads), got {tuple(alpha.shape)}"
            )
        if total_seqlen != q.shape[0]:
            raise RuntimeError(
                f"total_seqlen must match q.shape[0], got {total_seqlen} and {q.shape[0]}"
            )
        if cp_chunk_len % 64 != 0:
            raise RuntimeError(
                f"cp_chunk_len must be a multiple of 64, got {cp_chunk_len}"
            )
        if cu_seqlens.dtype != torch.int64:
            raise RuntimeError(
                f"cu_seqlens must have dtype torch.int64, got {cu_seqlens.dtype}"
            )
        if not cu_seqlens.is_contiguous():
            raise RuntimeError("cu_seqlens must be contiguous")
    num_seqs = cu_seqlens.shape[0] - 1
    if max_seqlen is None:
        raise RuntimeError("max_seqlen must be provided")
    if not _skip_check and max_seqlen <= 0:
        raise RuntimeError(f"max_seqlen must be positive, got {max_seqlen}")
    _, num_q_heads, d = q.shape
    num_k_heads = k.shape[1]
    num_v_heads = v.shape[1]
    num_sab_heads = max(num_q_heads, num_v_heads)
    total_t_blocks = workspace_num_chunks_host(cu_seqlens, 64, total_seqlen)
    total_cp_chunks = workspace_num_chunks_host(cu_seqlens, cp_chunk_len, total_seqlen)
    if not _skip_check:
        if o.shape[1] != num_sab_heads:
            raise RuntimeError(
                f"o heads must equal max(q heads, v heads)={num_sab_heads}, got {o.shape[1]}"
            )
        if alpha.shape[1] != num_sab_heads:
            raise RuntimeError(
                f"alpha heads must equal max(q heads, v heads)={num_sab_heads}, got {alpha.shape[1]}"
            )
        if not _FullyFusedDeltaRuleSm90.can_implement(
            num_q_heads, num_k_heads, num_v_heads, d, q.element_size()
        ):
            raise RuntimeError(
                "CPDeltaRulePrefillSm90 only supports head counts where q/v heads are positive multiples "
                f"of k heads, got q={num_q_heads}, k={num_k_heads}, v={num_v_heads}"
            )
        if t.shape != (total_t_blocks, num_sab_heads, 64, 64):
            raise RuntimeError(
                "t must have shape "
                f"{(total_t_blocks, num_sab_heads, 64, 64)}, got {tuple(t.shape)}"
            )
        expected_fixed_state_shape = (total_cp_chunks, num_sab_heads, d, d)
        expected_state_shape = (num_seqs, num_sab_heads, d, d)
        if (
            fixed_state.shape != expected_fixed_state_shape
            or state.shape != expected_state_shape
        ):
            raise RuntimeError(
                "fixed_state/state must have shapes "
                f"{expected_fixed_state_shape} and {expected_state_shape}, "
                f"got {tuple(fixed_state.shape)} and {tuple(state.shape)}"
            )
        if initial_state is not None and initial_state.shape != expected_state_shape:
            raise RuntimeError(
                f"initial_state must have shape {expected_state_shape}, got {tuple(initial_state.shape)}"
            )
        if q.shape[-1] != 128:
            raise RuntimeError(
                f"CPDeltaRulePrefillSm90 only supports D=128, got {q.shape[-1]}"
            )
        if q.dtype not in (torch.float16, torch.bfloat16):
            raise RuntimeError(
                f"CPDeltaRulePrefillSm90 only supports fp16/bf16 inputs, got {q.dtype}"
            )
        if (
            k.dtype != q.dtype
            or v.dtype != q.dtype
            or o.dtype != q.dtype
            or t.dtype != q.dtype
        ):
            raise RuntimeError(
                "q/k/v/o/t dtypes must match, "
                f"got q={q.dtype}, k={k.dtype}, v={v.dtype}, o={o.dtype}, t={t.dtype}"
            )
        if alpha.dtype != torch.float32:
            raise RuntimeError(
                f"alpha must have dtype torch.float32, got {alpha.dtype}"
            )
        if initial_state is not None and initial_state.dtype != torch.float32:
            raise RuntimeError(
                f"initial_state must have dtype torch.float32, got {initial_state.dtype}"
            )
        for name, tensor in (
            ("q", q),
            ("k", k),
            ("v", v),
            ("t", t),
            ("fixed_state", fixed_state),
            ("initial_state", initial_state),
            ("alpha", alpha),
            ("o", o),
            ("state", state),
        ):
            if tensor is None:
                continue
            if not tensor.is_contiguous():
                raise RuntimeError(f"{name} must be contiguous")
    max_cp_chunks_per_seq = max_num_chunks_host(max_seqlen, cp_chunk_len)
    if total_cp_chunks == 0:
        return
    q_tma = q.as_strided(
        (total_seqlen, d, num_q_heads),
        (num_q_heads * d, 1, d),
    )
    k_tma = k.as_strided(
        (d, total_seqlen, num_k_heads),
        (1, num_k_heads * d, d),
    )
    v_tma = v.as_strided(
        (d, total_seqlen, num_v_heads),
        (1, num_v_heads * d, d),
    )
    o_tma = o.as_strided(
        (d, total_seqlen, num_sab_heads),
        (1, num_sab_heads * d, d),
    )
    t_tma = t.as_strided(
        (64, 64, num_sab_heads, total_t_blocks),
        (
            64,
            1,
            64 * 64,
            num_sab_heads * 64 * 64,
        ),
    )

    kernel_dtype = {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
    }[q.dtype]

    workspace_size = get_device_sm_count(q.device) * 128
    tensormaps_t = _get_cache_buf("gdn_cp_prefill_tensormaps", workspace_size, q.device)
    stream_val = torch.cuda.current_stream().cuda_stream
    stream = cuda_driver.CUstream(stream_val)

    enable_tvm_ffi = True
    if enable_tvm_ffi:
        from_dlpack = lambda *args, **kwargs: cute.runtime.from_dlpack(
            *args, **{**kwargs, "enable_tvm_ffi": True}
        )

    needs_initial_state = initial_state is not None
    initial_state_cute = (
        from_dlpack(initial_state.reshape(-1), assumed_align=16).mark_layout_dynamic()
        if needs_initial_state
        else None
    )
    kernel = CPDeltaRulePrefillSm90(
        kernel_dtype, needs_initial_state=needs_initial_state
    )
    kernel_args = (
        from_dlpack(q_tma, assumed_align=16).mark_layout_dynamic(leading_dim=1),
        from_dlpack(k_tma, assumed_align=16).mark_layout_dynamic(leading_dim=0),
        from_dlpack(v_tma, assumed_align=16).mark_layout_dynamic(leading_dim=0),
        from_dlpack(t_tma, assumed_align=16).mark_layout_dynamic(leading_dim=1),
        from_dlpack(o_tma, assumed_align=16).mark_layout_dynamic(leading_dim=0),
        from_dlpack(alpha.reshape(-1), assumed_align=16).mark_layout_dynamic(),
        from_dlpack(state.reshape(-1), assumed_align=16).mark_layout_dynamic(),
        from_dlpack(fixed_state.reshape(-1), assumed_align=16).mark_layout_dynamic(),
        initial_state_cute,
        from_dlpack(tensormaps_t, assumed_align=128).mark_layout_dynamic(),
        from_dlpack(cu_seqlens, assumed_align=8).mark_layout_dynamic(),
        cutlass.Float32(scale),
        cutlass.Int32(num_q_heads),
        cutlass.Int32(num_k_heads),
        cutlass.Int32(num_v_heads),
        cutlass.Int32(num_sab_heads),
        cutlass.Int32(cp_chunk_len),
        cutlass.Int32(total_cp_chunks),
        cutlass.Int32(max_cp_chunks_per_seq),
        cutlass.Int32(num_seqs),
        stream,
    )
    compiled = cached_compile(
        kernel,
        *kernel_args,
        compile_options=(cute.GPUArch("sm_90a"),),
    )
    compiled(*kernel_args)


def cp_delta_rule_dsl_sm90(
    o: torch.Tensor,
    state: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    *,
    initial_state: torch.Tensor | None = None,
    max_seqlen: int | None = None,
    cp_chunk_len: int | None = None,
    cp_chunk_len_granularity: int = CP_CHUNK_LEN_GRANULARITY,
):
    """Run the CP SM90 delta-rule prefill pipeline on flat varlen tensors.

    Q/K/V/alpha/beta/O/state follow the same public layout as the non-CP
    prefill path. Internal T, M, N, and fixed-state tensors use the varlen
    workspace layout.
    """
    total_seqlen = q.shape[0]
    num_seqs = cu_seqlens.shape[0] - 1
    if max_seqlen is None and num_seqs == 1:
        max_seqlen = total_seqlen
    if max_seqlen is None:
        raise RuntimeError("max_seqlen must be provided when num_seqs != 1")
    if max_seqlen <= 0:
        raise RuntimeError(f"max_seqlen must be positive, got {max_seqlen}")
    if cp_chunk_len is None:
        num_heads = max(q.shape[1], v.shape[1])
        num_sms = get_device_sm_count(q.device)
        cp_chunk_len = choose_cp_chunk_len_host(
            max_seqlen,
            num_heads,
            num_sms,
            chunk_len_granularity=cp_chunk_len_granularity,
        )
    if q.ndim != 3:
        raise RuntimeError(
            f"q must have shape (total_seqlen, num_q_heads, D), got {tuple(q.shape)}"
        )
    if k.ndim != 3 or v.ndim != 3 or o.ndim != 3:
        raise RuntimeError(
            "k, v, and o must have shape (total_seqlen, num_heads, D), "
            f"got k={tuple(k.shape)}, v={tuple(v.shape)}, o={tuple(o.shape)}"
        )
    if k.shape[0] != q.shape[0] or v.shape[0] != q.shape[0] or o.shape[0] != q.shape[0]:
        raise RuntimeError("q, k, v, and o must have the same total_seqlen")
    if k.shape[2] != q.shape[2] or v.shape[2] != q.shape[2] or o.shape[2] != q.shape[2]:
        raise RuntimeError("q, k, v, and o must have the same D")
    if alpha.ndim != 2 or alpha.shape[0] != q.shape[0]:
        raise RuntimeError(
            f"alpha must have shape (total_seqlen, num_sab_heads), got {tuple(alpha.shape)}"
        )
    if beta.ndim != 2 or beta.shape[0] != q.shape[0]:
        raise RuntimeError(
            f"beta must have shape (total_seqlen, num_sab_heads), got {tuple(beta.shape)}"
        )
    if cp_chunk_len % 64 != 0:
        raise RuntimeError(f"cp_chunk_len must be a multiple of 64, got {cp_chunk_len}")
    if cu_seqlens.dtype != torch.int64:
        raise RuntimeError(
            f"cu_seqlens must have dtype torch.int64, got {cu_seqlens.dtype}"
        )
    if not cu_seqlens.is_contiguous():
        raise RuntimeError("cu_seqlens must be contiguous")
    _, num_q_heads, d = q.shape
    num_k_heads = k.shape[1]
    num_v_heads = v.shape[1]
    num_sab_heads = max(num_q_heads, num_v_heads)
    if o.shape[1] != num_sab_heads:
        raise RuntimeError(
            f"o heads must equal max(q heads, v heads)={num_sab_heads}, got {o.shape[1]}"
        )
    if alpha.shape[1] != num_sab_heads:
        raise RuntimeError(
            f"alpha heads must equal max(q heads, v heads)={num_sab_heads}, got {alpha.shape[1]}"
        )
    if beta.shape[1] != num_sab_heads:
        raise RuntimeError(
            f"beta heads must equal max(q heads, v heads)={num_sab_heads}, got {beta.shape[1]}"
        )
    if not _FullyFusedDeltaRuleSm90.can_implement(
        num_q_heads, num_k_heads, num_v_heads, d, q.element_size()
    ):
        raise RuntimeError(
            "CPDeltaRuleSm90 only supports head counts where q/v heads are positive multiples "
            f"of k heads, got q={num_q_heads}, k={num_k_heads}, v={num_v_heads}"
        )
    expected_state_shape = (num_seqs, num_sab_heads, d, d)
    if state.shape != expected_state_shape:
        raise RuntimeError(
            f"state must have shape {expected_state_shape}, got {tuple(state.shape)}"
        )
    if initial_state is not None and initial_state.shape != expected_state_shape:
        raise RuntimeError(
            f"initial_state must have shape {expected_state_shape}, got {tuple(initial_state.shape)}"
        )
    if q.shape[-1] != 128:
        raise RuntimeError(f"CPDeltaRuleSm90 only supports D=128, got {q.shape[-1]}")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(
            f"CPDeltaRuleSm90 only supports fp16/bf16 inputs, got {q.dtype}"
        )
    if k.dtype != q.dtype or v.dtype != q.dtype or o.dtype != q.dtype:
        raise RuntimeError(
            "q/k/v/o dtypes must match, "
            f"got q={q.dtype}, k={k.dtype}, v={v.dtype}, o={o.dtype}"
        )
    if alpha.dtype != torch.float32 or beta.dtype != torch.float32:
        raise RuntimeError(
            f"alpha/beta must have dtype torch.float32, got {alpha.dtype} and {beta.dtype}"
        )
    if initial_state is not None and initial_state.dtype != torch.float32:
        raise RuntimeError(
            f"initial_state must have dtype torch.float32, got {initial_state.dtype}"
        )
    for name, tensor in (
        ("q", q),
        ("k", k),
        ("v", v),
        ("alpha", alpha),
        ("beta", beta),
        ("o", o),
        ("state", state),
        ("initial_state", initial_state),
    ):
        if tensor is None:
            continue
        if not tensor.is_contiguous():
            raise RuntimeError(f"{name} must be contiguous")

    t = cp_delta_rule_t_precompute_dsl_sm90(
        k, beta, cu_seqlens, total_seqlen, max_seqlen=max_seqlen, _skip_check=True
    )
    local_transfer, local_state = cp_delta_rule_mn_precompute_dsl_sm90(
        k,
        v,
        t,
        alpha,
        cu_seqlens,
        total_seqlen,
        cp_chunk_len=cp_chunk_len,
        max_seqlen=max_seqlen,
        _skip_check=True,
    )
    fixed_state = cp_delta_rule_fixup_dsl_sm90(
        local_transfer,
        local_state,
        cu_seqlens,
        total_seqlen,
        cp_chunk_len=cp_chunk_len,
        initial_state=initial_state,
        _skip_check=True,
    )

    if num_seqs != 1:
        state.zero_()
    cp_delta_rule_prefill_dsl_sm90(
        o,
        state,
        q,
        k,
        v,
        t,
        fixed_state,
        alpha,
        scale,
        cu_seqlens,
        total_seqlen,
        cp_chunk_len=cp_chunk_len,
        max_seqlen=max_seqlen,
        initial_state=initial_state,
        _skip_check=True,
    )
