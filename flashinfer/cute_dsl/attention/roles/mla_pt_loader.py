# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""MLAPageTableLoaderRole — page-table producer warp for MLA decode.

Owns the tile-scheduler loop for the page-table warp, issuing async copies
of page indices from global memory into SMEM for each k-tile.
"""

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.cpasync as cpasync
from cutlass.pipeline import PipelineProducer

from ..mla_config import MLAConfig
from ..scheduler.mla_persistent import (
    create_mla_static_tile_scheduler,
    MLAStaticTileSchedulerParams,
    ceil_div,
)


class MLAPageTableLoaderRole:
    def __init__(self, config: MLAConfig):
        self.mma_qk_tiler = config.mma_qk_tiler
        self.page_size = config.page_size
        self.is_var_split_kv = config.is_var_split_kv
        self.load_pt_stage = config.load_pt_stage
        self.threads_per_warp = 32

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

    @cute.jit
    def run(
        self,
        split_kv: cutlass.Int32,
        cache_seqs: cute.Tensor,
        block_split_kvs: cute.Tensor,
        load_pt_producer: PipelineProducer,
        mPT: cute.Tensor,
        sPT: cute.Tensor,
        tile_sched_params: MLAStaticTileSchedulerParams,
    ):
        tile_sched = create_mla_static_tile_scheduler(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()

        tidx, _, _ = cute.arch.thread_idx()
        tidx = tidx % self.threads_per_warp
        page_per_tile = self.mma_qk_tiler[1] // self.page_size
        elem_per_thread = ceil_div(page_per_tile, self.threads_per_warp)

        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.ALWAYS),
            cutlass.Int32,
            num_bits_per_copy=cutlass.Int32.width,
        )

        while work_tile.is_valid_tile:
            blk_coord = work_tile.tile_idx
            k_index, k_tile_count, _ = self._get_k_tile_count(
                split_kv, cache_seqs, block_split_kvs, blk_coord
            )

            mPT_seq = mPT[None, blk_coord[2]]
            mPT_for_copy = cute.flat_divide(mPT_seq, (1,))
            sPT_for_copy = cute.flat_divide(sPT, (1,))

            while k_tile_count > 0:
                handle = load_pt_producer.acquire_and_advance()

                for i in range(elem_per_thread):
                    idx = i * self.threads_per_warp + tidx
                    if cute.elem_less(
                        k_index * page_per_tile + idx, mPT_seq.shape[0]
                    ) and cute.elem_less(idx, page_per_tile):
                        cute.copy(
                            atom_async_copy,
                            mPT_for_copy[None, k_index * page_per_tile + idx],
                            sPT_for_copy[None, idx, handle.index],
                        )
                    else:
                        sPT_for_copy[None, idx, handle.index].fill(0)

                handle.commit()
                k_index += 1
                k_tile_count -= 1

            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()

        load_pt_producer.tail()
