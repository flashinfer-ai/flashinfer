# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""PageTableLoaderRole — page-table producer warp for paged prefill.

Runs on the schedule's otherwise-idle empty warp when
``config.load_pt_stages`` is set (page_size 8: 16+ TMA copies per KV
tile).  Owns its own tile-scheduler loop — it re-derives the exact
work-item/KV-tile sequence the loader walks, with no handshake beyond
the ``load_pt`` pipeline — and per KV tile stages that tile's page ids
into an SMEM ring, one id per lane via ``cp.async``.  The loader then
reads ids back at LDS latency instead of paying a global-latency load
per page inside its TMA-issue loop (the MLA-decode pt-loader design).

Clamping happens here, at produce time, because both clamps are
positional — they depend only on the logical page index, never on the
table value: pages past the sequence end, and (with window_left) pages
below the work item's window lower bound, publish -1 instead of an id.
A -1 coordinate makes the TMA copy an OOB zero-fill, which the NaN
contract requires (see ``LoaderRole._load_paged_tile``).  The window
bound is applied on every tile, not just the first: later tiles' pages
all sit above it by construction (``page_idx_lb`` falls inside the
first tile's page range), so the comparison is vacuously true there and
the loader-side first-tile special case disappears.
"""

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.cpasync as cpasync
from cutlass.cute.typing import Int32
from cutlass.pipeline import PipelineProducer

from ..config import AttentionConfig
from ..fusion.mask import get_kv_block_range
from ..scheduler.persistent import (
    FmhaStaticTileScheduler,
    FmhaStaticTileSchedulerParams,
    create_fmha_static_tile_scheduler,
)


class PageTableLoaderRole:
    """Empty-warp producer that stages pre-clamped page ids for the loader."""

    def __init__(self, config: AttentionConfig):
        self.cta_tiler = config.cta_tiler
        self.mask_spec = config.mask_spec
        self.page_size = config.page_size
        self.pages_per_kv_tile = config.pages_per_kv_tile
        self.threads_per_warp = 32

    @cute.jit
    def run(
        self,
        seqlen_q_static: Int32,
        seqlen_k_static: Int32,
        cum_seqlen_q: cute.Tensor | None,
        cum_seqlen_k: cute.Tensor | None,
        window_left: Int32,
        window_right: Int32,
        load_pt_producer: PipelineProducer,
        kv_page_table: cute.Tensor,
        kv_page_indptr: cute.Tensor,
        sPT: cute.Tensor,
        tile_sched_params: FmhaStaticTileSchedulerParams,
    ):
        """Producer loop: one SMEM ring stage of clamped ids per KV tile.

        The work-item walk (scheduler advance, the short-seqlen_q skip,
        and the per-item KV block range) must match the loader's exactly
        — every stage produced here is consumed once by the loader's
        wait/release pair for the same tile.
        """
        tile_sched = create_fmha_static_tile_scheduler(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()

        tidx, _, _ = cute.arch.thread_idx()
        lane = tidx % self.threads_per_warp
        ids_per_lane = (
            self.pages_per_kv_tile + self.threads_per_warp - 1
        ) // self.threads_per_warp

        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.ALWAYS),
            cutlass.Int32,
            num_bits_per_copy=cutlass.Int32.width,
        )
        mPT_for_copy = cute.flat_divide(kv_page_table, (1,))
        sPT_for_copy = cute.flat_divide(sPT, (1,))

        while work_tile.is_valid_tile:
            curr_block_coord = work_tile.tile_idx
            batch_coord = curr_block_coord[2][1]
            continue_cond = False
            seqlen_q = seqlen_q_static
            if cutlass.const_expr(cum_seqlen_q is not None):
                cuseqlen_q = cum_seqlen_q[batch_coord]
                seqlen_q = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
                continue_cond = (
                    not FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                        self.cta_tiler[0],
                        curr_block_coord[0],
                        seqlen_q,
                    )
                )
            if not continue_cond:
                seqlen_k = seqlen_k_static
                if cutlass.const_expr(cum_seqlen_k is not None):
                    seqlen_k = cum_seqlen_k[batch_coord + 1] - cum_seqlen_k[batch_coord]

                table_base = kv_page_indptr[batch_coord]
                num_pages_kv = cute.ceil_div(seqlen_k, self.page_size)
                page_idx_lb = Int32(0)
                if cutlass.const_expr(self.mask_spec.has_window_left):
                    q0 = curr_block_coord[0] * self.cta_tiler[0]
                    lo_min = q0 + (seqlen_k - seqlen_q) - window_left
                    page_idx_lb = cutlass.max(lo_min, Int32(0)) // self.page_size

                kv_block_start, kv_block_end = get_kv_block_range(
                    self.mask_spec,
                    curr_block_coord,
                    self.cta_tiler,
                    seqlen_k,
                    seqlen_q,
                    window_left,
                    window_right,
                )
                kv_coord = kv_block_start
                pt_loop_steps = kv_block_end - kv_block_start
                for _t in cutlass.range(0, pt_loop_steps, 1, unroll=1):
                    handle = load_pt_producer.acquire_and_advance()
                    for i in cutlass.range_constexpr(ids_per_lane):
                        slot = i * self.threads_per_warp + lane
                        logical_page = kv_coord * self.pages_per_kv_tile + slot
                        in_range = logical_page < num_pages_kv
                        if cutlass.const_expr(self.mask_spec.has_window_left):
                            in_range = in_range & (logical_page >= page_idx_lb)
                        if cute.elem_less(slot, self.pages_per_kv_tile):
                            if in_range:
                                cute.copy(
                                    atom_async_copy,
                                    mPT_for_copy[None, table_base + logical_page],
                                    sPT_for_copy[None, slot, handle.index],
                                )
                            else:
                                sPT_for_copy[None, slot, handle.index].fill(-1)
                    handle.commit()
                    kv_coord += 1

            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()

        load_pt_producer.tail()
