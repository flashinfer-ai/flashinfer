# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""MLAMmaFP8Role — MMA warp role for FP8 MLA decode attention kernels.

FP8 differs from FP16 in three structural ways:

1. QK GEMM: single load_k wait covers all latent+rope stages, single release.
   K-rope uses separate tSrKC_rope fragments from sKC_rope SMEM.
   Fragment indexing: tSrQ[..., (q_stage, 0)], tSrKC[..., (q_stage, kc_stage)].

2. PV GEMM: loop order is ``for acc_stage -> mma_o_acquire -> for p_stage``.
   One load_v wait covers all p_stage iterations; mma_o produced per acc_stage.
   V fragment indexing: tOrVC[..., ((acc_stage, p_stage), vc_stage)].

3. mma_o pipeline: 2 stages (vs 1 for FP16), with acquire/commit per acc_stage.

GEMM helpers use explicit ``accumulate`` parameter and ``cutlass.range()``
(not ``range_constexpr``) for inner k-block loops, matching the FP16 pattern
in mla_mma.py. ``range()`` generates ``scf.for`` which keeps ``.set()`` SSA
values properly scoped; ``range_constexpr`` unrolls at compile time and leaks
SSA values across dynamic loop boundaries.
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


class MLAMmaFP8Role:
    """MMA warp for FP8 MLA decode — computes QK and PV GEMMs in TMEM.

    Does NOT own TMEM alloc/dealloc (that stays in the kernel for coordination).
    """

    def __init__(self, config: MLAConfig, mainloop: MLAMainloopSpec):
        self.mma_qk_tiler = config.mma_qk_tiler
        self.mma_qk_rope_tiler = config.mma_qk_rope_tiler
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

    @cute.jit
    def _get_k_tile_count(
        self,
        split_kv: cutlass.Int32,
        cache_seqs: cute.Tensor,
        block_split_kvs: cute.Tensor,
        blk_coord: cute.Coord,
    ) -> tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32]:
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
                qk_params.tSrQ[None, None, k_block, (q_stage, 0)],
                qk_params.tSrKC[None, None, k_block, (q_stage, kv_stage_index)],
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
        """Compute one QK-rope stage using separate tSrKC_rope fragments."""
        tStS = qk_params.tStS_staged[None, None, None, s_stage_index]
        for k_block in cutlass.range(self.rope_dim // tiled_mma_qk.shape_mnk[2]):
            tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, k_block != 0 or accumulate)
            cute.gemm(
                tiled_mma_qk,
                tStS,
                qk_params.tSrQ_rope[None, None, k_block, q_stage],
                qk_params.tSrKC_rope[None, None, k_block, kv_stage_index],
                tStS,
            )

    @cute.jit
    def _gemm_pv_one_stage(
        self,
        pv_params: SimpleNamespace,
        tiled_mma_pv: cute.TiledMma,
        p_stage_index: cutlass.Int32,
        vc_stage_index: cutlass.Int32,
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
                pv_params.tOrVC[
                    None, None, k_block, ((acc_stage, p_stage), vc_stage_index)
                ],
                tOtO,
            )

    # ------------------------------------------------------------------
    #  Orchestration loop
    # ------------------------------------------------------------------

    @cute.jit
    def run(
        self,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        load_q_consumer: PipelineConsumer,
        load_k_consumer: PipelineConsumer,
        load_v_consumer: PipelineConsumer,
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
        sKC_rope: cute.Tensor,
        sP: cute.Tensor,
        sVC: cute.Tensor,
        tmem_ptr: cute.Tensor,
        is_leader_cta: cutlass.Boolean,
        L: cutlass.Int32,
    ):
        """MMA warp orchestration for FP8 MLA decode."""
        tSrQ = tiled_mma_qk.make_fragment_A(sQ)
        tSrQ_rope = tiled_mma_qk.make_fragment_A(sQ_rope)
        tSrKC = tiled_mma_qk.make_fragment_B(sKC)
        tSrKC_rope = tiled_mma_qk.make_fragment_B(sKC_rope)
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

        tile_sched = create_mla_static_tile_scheduler(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()

        # Track PV accumulate state as a plain bool instead of
        # tiled_mma_pv.set/get to avoid carrying TiledMma SSA values
        # across the dynamic while-loop yield.
        pv_accumulated = False

        while work_tile.is_valid_tile:
            pv_accumulated = False
            blk_coord = work_tile.tile_idx
            k_index, k_tile_count, local_split_kv = self._get_k_tile_count(
                split_kv, cache_seqs, block_split_kvs, blk_coord
            )
            if k_tile_count > 0:
                mma_qk_params = SimpleNamespace(
                    sQ=sQ,
                    sQ_rope=sQ_rope,
                    sKC=sKC,
                    sKC_rope=sKC_rope,
                    tSrQ=tSrQ,
                    tSrQ_rope=tSrQ_rope,
                    tSrKC=tSrKC,
                    tSrKC_rope=tSrKC_rope,
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
                    # === First QK tile (with Q wait) ===
                    q_handle = load_q_consumer.wait_and_advance()

                    s_handle = mma_s_producer.acquire_and_advance()
                    kv_handle = load_k_consumer.wait_and_advance()
                    for q_stage in range(self.iterations_qk_latent):
                        self._gemm_qk_latent_one_stage(
                            mma_qk_params,
                            tiled_mma_qk,
                            s_handle.index,
                            kv_handle.index,
                            q_stage,
                            accumulate=(q_stage > 0),
                        )
                    for q_stage in range(self.iterations_qk_rope):
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
                        kv_handle = load_k_consumer.wait_and_advance()
                        for q_stage in range(self.iterations_qk_latent):
                            self._gemm_qk_latent_one_stage(
                                mma_qk_params,
                                tiled_mma_qk,
                                s_handle.index,
                                kv_handle.index,
                                q_stage,
                                accumulate=(q_stage > 0),
                            )
                        for q_stage in range(self.iterations_qk_rope):
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

                        # PV
                        p_handle = p_mma_consumer.wait_and_advance()
                        v_handle = load_v_consumer.wait_and_advance()
                        for acc_stage in range(self.iterations_pv_n):
                            o_handle = mma_o_producer.acquire_and_advance()
                            for p_stage in range(self.iterations_pv_k):
                                self._gemm_pv_one_stage(
                                    mma_pv_params,
                                    tiled_mma_pv,
                                    p_handle.index,
                                    v_handle.index,
                                    p_stage,
                                    acc_stage,
                                    accumulate=(pv_accumulated or p_stage > 0),
                                )
                            o_handle.commit()
                        v_handle.release()
                        p_handle.release()
                        pv_accumulated = True

                        k_tile_count -= 1

                    q_handle.release()

                    # === Final PV tile ===
                    p_handle = p_mma_consumer.wait_and_advance()
                    v_handle = load_v_consumer.wait_and_advance()
                    for acc_stage in range(self.iterations_pv_n):
                        o_handle = mma_o_producer.acquire_and_advance()
                        for p_stage in range(self.iterations_pv_k):
                            self._gemm_pv_one_stage(
                                mma_pv_params,
                                tiled_mma_pv,
                                p_handle.index,
                                v_handle.index,
                                p_stage,
                                acc_stage,
                                accumulate=(pv_accumulated or p_stage > 0),
                            )
                        o_handle.commit()
                    v_handle.release()
                    p_handle.release()

            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()

        mma_s_producer.tail()
        mma_o_producer.tail()
