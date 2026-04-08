# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""MLAMmaRole — MMA warp role for MLA decode attention kernels.

Extracted from the monolithic mla_decode_fp16.py kernel. Owns:
- Fragment creation for QK and PV GEMMs
- Per-stage GEMM helpers for QK latent/rope and PV
- run(): tile scheduler loop with interleaved QK/PV, pipeline lifecycle

All pipeline acquire/wait/commit/release/tail calls happen directly in run(),
not in sub-methods, because CuTe DSL compiles Python to MLIR/SSA where the
JIT boundary acts as pass-by-value for DSL metadata (TiledMma fields,
PipelineState). Mutations made inside @cute.jit sub-methods create new SSA
values that are invisible to the caller.

GEMM helpers take an explicit ``accumulate`` parameter following the FMHA
prefill pattern (roles/mma.py:gemm_pv). The ACCUMULATE flag is set inside
each helper as ``k_block != 0 or accumulate``, making it deterministic from
the parameter and inner loop index. Callers compute the parameter from their
own loop position — never from ``tiled_mma.get()`` after a sub-method return.
"""

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
from cutlass.pipeline import PipelineProducer, PipelineConsumer
from types import SimpleNamespace

from ..mla_config import MLAConfig
from ..mainloop_spec import MLAMainloopSpec
from ..scheduler.mla_persistent import (
    create_mla_static_tile_scheduler,
    MLAStaticTileSchedulerParams,
)


class MLAMmaRole:
    """MMA warp for MLA decode — computes QK and PV GEMMs in TMEM.

    Created from MLAConfig and MLAMainloopSpec in the kernel's __init__.
    Does NOT own TMEM alloc/dealloc (that stays in the kernel for coordination).
    """

    def __init__(self, config: MLAConfig, mainloop: MLAMainloopSpec):
        self.mma_qk_tiler = config.mma_qk_tiler
        self.mma_pv_tiler = config.mma_pv_tiler
        self.rope_dim = config.rope_dim
        self.latent_dim = config.latent_dim
        self.warps_in_n = config.warps_in_n
        self.mma_s_stage = config.mma_s_stage
        self.tmem_o_offset = config.tmem_o_offset
        self.iterations_qk_latent = config.iterations_qk_latent
        self.iterations_qk_rope = config.iterations_qk_rope
        self.iterations_pv_k = config.iterations_pv_k
        self.iterations_pv_n = config.iterations_pv_n
        self.enable_pdl = config.enable_pdl
        self.is_var_split_kv = config.is_var_split_kv

    # ------------------------------------------------------------------
    #  Tile count
    # ------------------------------------------------------------------

    @cute.jit
    def _get_k_tile_count(
        self,
        split_kv: cutlass.Int32,
        cache_seqs: cute.Tensor,
        block_split_kvs: cute.Tensor,
        blk_coord: cute.Coord,
    ) -> tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32]:
        """Get k_index, k_tile_count, and local split_kv for an MLA work tile.

        :param split_kv: Split_kv value
        :param cache_seqs: Cache sequence lengths tensor
        :param block_split_kvs: Per-block split_kv values tensor
        :param blk_coord: Block coordinate
        :return: k_index, k_tile_count, split_kv
        """
        K = cache_seqs[blk_coord[2]]
        if cutlass.const_expr(self.is_var_split_kv):
            split_kv = block_split_kvs[blk_coord[2]]

        k_tile_total = cute.ceil_div(K, self.mma_qk_tiler[1])
        k_tile_per_cta = cute.ceil_div(k_tile_total, split_kv)
        k_index = blk_coord[3] * k_tile_per_cta
        k_tile_count = max(0, min(k_tile_total, k_index + k_tile_per_cta) - k_index)
        return k_index, k_tile_count, split_kv

    # ------------------------------------------------------------------
    #  GEMM helpers — stateless w.r.t. caller
    #
    #  Each helper takes an explicit ``accumulate`` bool that controls
    #  whether the first k-block overwrites (False) or accumulates (True).
    #  Subsequent k-blocks always accumulate.  The caller computes the
    #  flag from its own loop position; the helper never communicates
    #  state back via TiledMma mutations (they would be invisible to the
    #  caller due to SSA pass-by-value at the @cute.jit boundary).
    #
    #  Inner k-block loops use ``cutlass.range()`` (dynamic scf.for),
    #  NOT ``cutlass.range_constexpr()`` (compile-time unroll).
    #  range_constexpr unrolls tiled_mma.set() calls into the enclosing
    #  scope, producing SSA values that leak across dynamic while-loop
    #  yields. range() keeps the .set() inside an scf.for scope where
    #  SSA carry-through is handled correctly.
    # ------------------------------------------------------------------

    @cute.jit
    def _gemm_qk_latent_one_stage(
        self,
        qk_params: SimpleNamespace,
        tiled_mma_qk: cute.TiledMma,
        s_stage_index: cutlass.Int32,
        kv_stage_index: cutlass.Int32,
        q_stage: int,
        accumulate: bool,
    ):
        """Compute one QK-latent stage: inner k-block GEMM loop."""
        tStS = qk_params.tStS_staged[None, None, None, s_stage_index]
        for k_block in cutlass.range(cute.size(qk_params.tSrQ.shape[2])):
            tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, k_block != 0 or accumulate)
            cute.gemm(
                tiled_mma_qk,
                tStS,
                qk_params.tSrQ[None, None, k_block, q_stage],
                qk_params.tSrKC[None, None, k_block, kv_stage_index],
                tStS,
            )

    @cute.jit
    def _gemm_qk_rope_one_stage(
        self,
        qk_params: SimpleNamespace,
        tiled_mma_qk: cute.TiledMma,
        s_stage_index: cutlass.Int32,
        kv_stage_index: cutlass.Int32,
        q_stage: int,
        accumulate: bool,
    ):
        """Compute one QK-rope stage: inner k-block GEMM loop."""
        tStS = qk_params.tStS_staged[None, None, None, s_stage_index]
        for k_block in cutlass.range(self.rope_dim // tiled_mma_qk.shape_mnk[2]):
            tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, k_block != 0 or accumulate)
            cute.gemm(
                tiled_mma_qk,
                tStS,
                qk_params.tSrQ_rope[None, None, k_block, q_stage],
                qk_params.tSrKC[None, None, k_block, kv_stage_index],
                tStS,
            )

    @cute.jit
    def _gemm_pv_one_stage(
        self,
        pv_params: SimpleNamespace,
        tiled_mma_pv: cute.TiledMma,
        p_stage_index: cutlass.Int32,
        kv_stage_index: cutlass.Int32,
        p_stage: int,
        acc_stage: int,
        accumulate: bool,
    ):
        """Compute one PV stage: inner k-block GEMM loop."""
        tOtO = pv_params.tOtO_staged[None, None, None, acc_stage]
        for k_block in cutlass.range(pv_params.tOrP.shape[2]):
            tiled_mma_pv.set(tcgen05.Field.ACCUMULATE, k_block != 0 or accumulate)
            cute.gemm(
                tiled_mma_pv,
                tOtO,
                pv_params.tOrP[
                    None,
                    None,
                    k_block,
                    (p_stage, p_stage_index),
                ],
                pv_params.tOrVC[None, None, k_block, kv_stage_index],
                tOtO,
            )

    # ------------------------------------------------------------------
    #  Orchestration loop — tile scheduler + interleaved QK/PV
    #
    #  All pipeline acquire/wait/commit/release/tail calls live here.
    #  Sub-methods only receive stage indices and do pure GEMM computation.
    # ------------------------------------------------------------------

    @cute.jit
    def run(
        self,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        load_q_consumer: PipelineConsumer,
        load_kv_consumer: PipelineConsumer,
        mma_s_producer: PipelineProducer,
        p_mma_consumer: PipelineConsumer,
        mma_o_producer: PipelineProducer,
        split_kv: cutlass.Int32,
        cache_seqs: cute.Tensor,
        block_split_kvs: cute.Tensor,
        tile_sched_params: MLAStaticTileSchedulerParams,
        sQ: cute.Tensor,
        sQ_rope: cute.Tensor,
        sKC: cute.Tensor,
        sP: cute.Tensor,
        sVC: cute.Tensor,
        tmem_ptr: cute.Tensor,
        is_leader_cta: cutlass.Boolean,
        L: cutlass.Int32,
    ):
        """MMA warp orchestration loop for MLA decode.

        Creates MMA fragments, iterates over work tiles via the tile scheduler,
        and runs interleaved QK/PV GEMMs with pipeline synchronization.

        Does NOT own TMEM alloc/dealloc — that stays in the kernel for
        coordination with other warps.
        """
        # Create MMA fragments
        tSrQ = tiled_mma_qk.make_fragment_A(sQ)
        tSrQ_rope = tiled_mma_qk.make_fragment_A(sQ_rope)
        tSrKC = tiled_mma_qk.make_fragment_B(sKC)
        tOrP = tiled_mma_pv.make_fragment_A(sP)
        tOrVC = tiled_mma_pv.make_fragment_B(sVC)

        tStS_shape = tiled_mma_qk.partition_shape_C(
            cute.select(self.mma_qk_tiler, mode=[0, 1])
        )
        tStS_staged_fake = tiled_mma_qk.make_fragment_C(
            cute.append(tStS_shape, self.mma_s_stage)
        )
        tStS_staged = cute.make_tensor(tmem_ptr, tStS_staged_fake.layout)
        tOtO_shape = tiled_mma_pv.partition_shape_C(
            cute.select(self.mma_pv_tiler, mode=[0, 1])
        )
        tOtO = tiled_mma_pv.make_fragment_C(tOtO_shape)
        tOtO_layout = cute.append(
            tOtO.layout,
            cute.make_layout(
                L // self.mma_pv_tiler[1],
                stride=self.mma_pv_tiler[1] // self.warps_in_n,
            ),
        )
        tOtO_staged = cute.make_tensor(
            tStS_staged.iterator + self.tmem_o_offset, tOtO_layout
        )

        # Tile scheduler
        tile_sched = create_mla_static_tile_scheduler(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()

        while work_tile.is_valid_tile:
            # Reset PV accumulate for each new work tile
            tiled_mma_pv.set(tcgen05.Field.ACCUMULATE, False)
            blk_coord = work_tile.tile_idx
            k_index, k_tile_count, local_split_kv = self._get_k_tile_count(
                split_kv, cache_seqs, block_split_kvs, blk_coord
            )
            if k_tile_count > 0:
                mma_qk_params = SimpleNamespace(
                    sQ=sQ,
                    sQ_rope=sQ_rope,
                    sKC=sKC,
                    tSrQ=tSrQ,
                    tSrQ_rope=tSrQ_rope,
                    tSrKC=tSrKC,
                    tStS_staged=tStS_staged,
                )
                mma_pv_params = SimpleNamespace(
                    sP=sP,
                    sVC=sVC,
                    tOrP=tOrP,
                    tOrVC=tOrVC,
                    tOtO_staged=tOtO_staged,
                )

                if is_leader_cta:
                    q_handle = load_q_consumer.wait_and_advance()

                    # === First QK tile ===
                    s_handle = mma_s_producer.acquire_and_advance()
                    for q_stage in range(self.iterations_qk_latent):
                        kv_handle = load_kv_consumer.wait_and_advance()
                        self._gemm_qk_latent_one_stage(
                            mma_qk_params,
                            tiled_mma_qk,
                            s_handle.index,
                            kv_handle.index,
                            q_stage,
                            accumulate=(q_stage > 0),
                        )
                        kv_handle.release()
                    for q_stage in range(self.iterations_qk_rope):
                        kv_handle = load_kv_consumer.wait_and_advance()
                        self._gemm_qk_rope_one_stage(
                            mma_qk_params,
                            tiled_mma_qk,
                            s_handle.index,
                            kv_handle.index,
                            q_stage,
                            accumulate=True,
                        )
                        kv_handle.release()
                    s_handle.commit()
                    k_tile_count -= 1

                    # === Interleaved QK + PV for remaining tiles ===
                    while k_tile_count > 0:
                        # QK
                        s_handle = mma_s_producer.acquire_and_advance()
                        for q_stage in range(self.iterations_qk_latent):
                            kv_handle = load_kv_consumer.wait_and_advance()
                            self._gemm_qk_latent_one_stage(
                                mma_qk_params,
                                tiled_mma_qk,
                                s_handle.index,
                                kv_handle.index,
                                q_stage,
                                accumulate=(q_stage > 0),
                            )
                            kv_handle.release()
                        for q_stage in range(self.iterations_qk_rope):
                            kv_handle = load_kv_consumer.wait_and_advance()
                            self._gemm_qk_rope_one_stage(
                                mma_qk_params,
                                tiled_mma_qk,
                                s_handle.index,
                                kv_handle.index,
                                q_stage,
                                accumulate=True,
                            )
                            kv_handle.release()
                        s_handle.commit()

                        # PV — pv_acc is read at run() level (safe), tracks
                        # whether any PV block has already initialized TMEM O.
                        o_handle = mma_o_producer.acquire_and_advance()
                        p_handle = p_mma_consumer.wait_and_advance()
                        pv_acc = tiled_mma_pv.get(tcgen05.Field.ACCUMULATE)
                        for p_stage in range(self.iterations_pv_k):
                            for acc_stage in range(self.iterations_pv_n):
                                kv_handle = load_kv_consumer.wait_and_advance()
                                self._gemm_pv_one_stage(
                                    mma_pv_params,
                                    tiled_mma_pv,
                                    p_handle.index,
                                    kv_handle.index,
                                    p_stage,
                                    acc_stage,
                                    accumulate=(pv_acc or p_stage > 0),
                                )
                                kv_handle.release()
                        p_handle.release()
                        o_handle.commit()
                        tiled_mma_pv.set(tcgen05.Field.ACCUMULATE, True)

                        k_tile_count -= 1

                    q_handle.release()

                    # === Final PV tile ===
                    o_handle = mma_o_producer.acquire_and_advance()
                    p_handle = p_mma_consumer.wait_and_advance()
                    pv_acc = tiled_mma_pv.get(tcgen05.Field.ACCUMULATE)
                    for p_stage in range(self.iterations_pv_k):
                        for acc_stage in range(self.iterations_pv_n):
                            kv_handle = load_kv_consumer.wait_and_advance()
                            self._gemm_pv_one_stage(
                                mma_pv_params,
                                tiled_mma_pv,
                                p_handle.index,
                                kv_handle.index,
                                p_stage,
                                acc_stage,
                                accumulate=(pv_acc or p_stage > 0),
                            )
                            kv_handle.release()
                    p_handle.release()
                    o_handle.commit()

            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()

        # Pipeline producer tails
        mma_s_producer.tail()
        mma_o_producer.tail()
