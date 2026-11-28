# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Softmax tasks: online softmax over the S tiles and P production."""

import cutlass
from cutlass.experimental.task_scheduling.resources import MemoryResource, WorkQueue
from cutlass.experimental.task_scheduling.schedule_builder import domain_loop
from cutlass.experimental.task_scheduling.task import Task
from ..config import FmhaDecodeConfig
from .common import (
    DecodeGenTask,
    TaskKwarg,
    decode_work_tile_schedule,
    schedule_with_optional_resources,
)
from ....sage.modes import SageScoreScaling


def create_softmax0_task(
    tmem_s0: MemoryResource,
    tmem_softmax_local0: MemoryResource,
    smem_p0: MemoryResource,
    tmem_softmax_global0: MemoryResource,
    tmem_softmax_order: MemoryResource | None,
    sparse_softmax_metadata: MemoryResource | None,
    work_queue: WorkQueue | None,
    cfg: FmhaDecodeConfig,
    *,
    domain: int | cutlass.Int32,
    membership_lifetime: MemoryResource | None = None,
    sage_k_scales: MemoryResource | None = None,
    task_class: type[DecodeGenTask] = DecodeGenTask,
    **kw: TaskKwarg,
) -> Task:
    """Create the first softmax task, including optional ordered publication."""
    memberships_in_gmem = reads_memberships_from_gmem(tmem_s0)

    @schedule_with_optional_resources
    def softmax0_schedule(
        tmem_s0: MemoryResource,
        tmem_softmax_local0: MemoryResource,
        smem_p0: MemoryResource,
        tmem_softmax_global0: MemoryResource,
        tmem_softmax_order: MemoryResource | None,
        sparse_softmax_metadata: MemoryResource | None,
        membership_lifetime: MemoryResource | None,
        sage_k_scales: MemoryResource | None,
        work_queue: WorkQueue | None,
    ) -> None:
        """Schedule softmax0 with the supplied order and sparse resources."""
        decode_work_tile_schedule(
            cfg,
            work_queue,
            lambda: softmax_schedule_body(
                cfg,
                0,
                domain,
                tmem_s0,
                tmem_softmax_local0,
                smem_p0,
                tmem_softmax_global0,
                tmem_softmax_order,
                sparse_softmax_metadata,
                membership_lifetime,
                sage_k_scales,
                memberships_in_gmem=memberships_in_gmem,
            ),
        )

    schedule_result = softmax0_schedule(
        tmem_s0,
        tmem_softmax_local0,
        smem_p0,
        tmem_softmax_global0,
        tmem_softmax_order,
        sparse_softmax_metadata,
        membership_lifetime,
        sage_k_scales,
        work_queue,
    )
    src = [tmem_s0]
    if sparse_softmax_metadata is not None:
        src.append(sparse_softmax_metadata)
    if membership_lifetime is not None:
        src.append(membership_lifetime)
    if sage_k_scales is not None:
        src.append(sage_k_scales)
    if work_queue is not None:
        src.append(work_queue)
    dst = [tmem_softmax_local0, smem_p0, tmem_softmax_global0]
    if tmem_softmax_order is not None:
        dst.append(tmem_softmax_order)
    return task_class(
        src_resources=src,
        dst_resources=dst,
        q_bound_resources=((tmem_s0, False),),
        cfg=cfg,
        warp_idx=cfg.softmax0_warp_idx,
        num_warps=cfg.softmax0_num_warps,
        schedule=schedule_result,
        num_registers=cfg.softmax_task_num_registers,
        name="Softmax0Task",
        consumes_visible_kv_interval=True,
        **kw,
    )


def create_softmax1_task(
    tmem_s1: MemoryResource,
    tmem_softmax_local1: MemoryResource,
    smem_p1: MemoryResource,
    tmem_softmax_global1: MemoryResource,
    tmem_softmax_order: MemoryResource | None,
    sparse_softmax_metadata: MemoryResource | None,
    work_queue: WorkQueue | None,
    cfg: FmhaDecodeConfig,
    *,
    domain: int | cutlass.Int32,
    membership_lifetime: MemoryResource | None = None,
    sage_k_scales: MemoryResource | None = None,
    task_class: type[DecodeGenTask] = DecodeGenTask,
    **kw: TaskKwarg,
) -> Task:
    """Create the second softmax task, including optional ordered publication."""
    memberships_in_gmem = reads_memberships_from_gmem(tmem_s1)

    @schedule_with_optional_resources
    def softmax1_schedule(
        tmem_s1: MemoryResource,
        tmem_softmax_local1: MemoryResource,
        smem_p1: MemoryResource,
        tmem_softmax_global1: MemoryResource,
        tmem_softmax_order: MemoryResource | None,
        sparse_softmax_metadata: MemoryResource | None,
        membership_lifetime: MemoryResource | None,
        sage_k_scales: MemoryResource | None,
        work_queue: WorkQueue | None,
    ) -> None:
        """Schedule softmax1 with the supplied order and sparse resources."""
        # Prime the first P0 -> P1 baton once per CTA.  Each completed P1
        # publication leaves the same barrier half-arrived for the next
        # persistent work tile, so re-priming inside the work-tile loop would
        # toggle its phase early and deadlock on the second tile.
        if tmem_softmax_order is not None:
            tmem_softmax_order.prime_softmax1()
        decode_work_tile_schedule(
            cfg,
            work_queue,
            lambda: softmax_schedule_body(
                cfg,
                1,
                domain,
                tmem_s1,
                tmem_softmax_local1,
                smem_p1,
                tmem_softmax_global1,
                tmem_softmax_order,
                sparse_softmax_metadata,
                membership_lifetime,
                sage_k_scales,
                memberships_in_gmem=memberships_in_gmem,
            ),
        )

    schedule_result = softmax1_schedule(
        tmem_s1,
        tmem_softmax_local1,
        smem_p1,
        tmem_softmax_global1,
        tmem_softmax_order,
        sparse_softmax_metadata,
        membership_lifetime,
        sage_k_scales,
        work_queue,
    )
    src = [tmem_s1]
    if sparse_softmax_metadata is not None:
        src.append(sparse_softmax_metadata)
    if tmem_softmax_order is not None:
        src.append(tmem_softmax_order)
    if membership_lifetime is not None:
        src.append(membership_lifetime)
    if sage_k_scales is not None:
        src.append(sage_k_scales)
    if work_queue is not None:
        src.append(work_queue)
    dst = [tmem_softmax_local1, smem_p1, tmem_softmax_global1]
    return task_class(
        src_resources=src,
        dst_resources=dst,
        q_bound_resources=((tmem_s1, False),),
        cfg=cfg,
        warp_idx=cfg.softmax1_warp_idx,
        num_warps=cfg.softmax1_num_warps,
        schedule=schedule_result,
        num_registers=cfg.softmax_task_num_registers,
        name="Softmax1Task",
        consumes_visible_kv_interval=True,
        **kw,
    )


def reads_memberships_from_gmem(tmem_s: MemoryResource) -> bool:
    """Whether Softmax reads grouped-Q page memberships from GMEM."""
    page_offsets = getattr(tmem_s, "page_offsets_ref", None)
    return page_offsets is not None and not page_offsets.cache_memberships_in_smem


def softmax_schedule_body(
    cfg: FmhaDecodeConfig,
    inst_id: int,
    domain: int | cutlass.Int32,
    tmem_s: MemoryResource,
    tmem_softmax_local: MemoryResource,
    smem_p: MemoryResource,
    tmem_softmax_global: MemoryResource,
    tmem_softmax_order: MemoryResource | None,
    sparse_softmax_metadata: MemoryResource | None,
    membership_lifetime: MemoryResource | None,
    sage_k_scales: MemoryResource | None,
    memberships_in_gmem: bool = False,
) -> None:
    """Build one softmax instance's loop, P publication, and final stats handoff.

    Both instances run this body and differ only in their side of the P
    baton.
    """
    if membership_lifetime is not None:
        membership_lifetime.wait()
    (
        old_max_arr,
        sum_arr,
        new_max_arr,
        s_arr,
    ) = tmem_s.init_softmax_state()
    smem_p.init_compute_state()
    if sparse_softmax_metadata is not None:
        sparse_softmax_metadata.init_read_state()
    if cutlass.const_expr(cfg.use_sage_attention):
        # The lane's Q row is fixed for the work tile.
        sage_q_scale = tmem_s.load_sage_q_scale()
    # The work framework rejects ``None`` arguments, so each Sage/sparse
    # combination is its own work callable.
    use_sparse = sparse_softmax_metadata is not None
    use_sage = cfg.use_sage_attention
    # The ``sfK`` words sit in the instance's ring, which the load warp fills
    # with each tile. They are read in place, so the ring stage stays held
    # through the last pass that reads them: the max pass for dequantized
    # scores, else the P pass.
    releases_sage_stage_after_max = (
        cfg.sage_score_scaling() is SageScoreScaling.DEQUANTIZED
    )
    compute_softmax_loop = {
        (False, False): tmem_s.compute_softmax_loop,
        (False, True): tmem_s.compute_sage_softmax_loop,
        (True, False): tmem_s.compute_block_sparse_softmax_loop,
        (True, True): tmem_s.compute_sage_block_sparse_softmax_loop,
    }[(use_sparse, use_sage)]
    compute_p_fragments = {
        (False, False): smem_p.compute_p_fragments,
        (False, True): smem_p.compute_sage_p_fragments,
        (True, False): smem_p.compute_proxy_route_p_fragments,
        (True, True): smem_p.compute_sage_proxy_route_p_fragments,
    }[(cfg.use_block_sparse_proxy_routes, use_sage)]

    # GMEM-resident grouped-Q page memberships are independent of S; see
    # FmhaDecodeConfig.preloads_page_memberships.
    preloads_memberships = (
        cfg.preloads_page_memberships
        and not use_sparse
        and not use_sage
        and memberships_in_gmem
    )
    if cutlass.const_expr(preloads_memberships):
        page_membership_arr = tmem_s.load_first_page_memberships()
    with domain_loop(0, domain, 1, unroll=1) as d:
        if use_sparse:
            # Consume the independent metadata stream first so its SMEM
            # loads and release can overlap the subsequent score wait.
            sparse_softmax_metadata.wait()
            # Copy the complete payload to registers before release, so
            # masking cannot race the producer's next SMEM-stage reuse.
            (
                sparse_origin0,
                sparse_origin1,
                sparse_route_flags,
                sparse_token_word0,
                sparse_token_word1,
                sparse_token_word2,
                sparse_token_word3,
            ) = sparse_softmax_metadata.load_route()
            sparse_softmax_metadata.release()
        if cutlass.const_expr(use_sage):
            # ConsWait/ConsWork: the tile's ``sfK`` words; both passes read
            # the waited stage through its routed offset.
            sage_k_scales.wait()
            sage_k_stage_bytes = sage_k_scales.open_stage()
        # ConsWait/ConsWork: load S from TMEM and compute the tile max.
        tmem_s.wait()
        softmax_loop_kwargs = dict(
            old_max_arr=old_max_arr,
            sum_arr=sum_arr,
            new_max_arr=new_max_arr,
            s_arr=s_arr,
        )
        if cutlass.const_expr(preloads_memberships):
            compute_softmax_loop = tmem_s.compute_softmax_loop_with_memberships
            softmax_loop_kwargs["page_membership_arr"] = page_membership_arr
        if cutlass.const_expr(cfg.use_sage_attention):
            softmax_loop_kwargs["sage_q_scale"] = sage_q_scale
            softmax_loop_kwargs["sage_k_stage_bytes"] = sage_k_stage_bytes
        if use_sparse:
            softmax_loop_kwargs.update(
                sparse_origin0=sparse_origin0,
                sparse_origin1=sparse_origin1,
                sparse_route_flags=sparse_route_flags,
                sparse_token_word0=sparse_token_word0,
                sparse_token_word1=sparse_token_word1,
                sparse_token_word2=sparse_token_word2,
                sparse_token_word3=sparse_token_word3,
            )
        old_max_arr, sum_arr, new_max_arr, s_arr = compute_softmax_loop(
            **softmax_loop_kwargs
        )
        if cutlass.const_expr(preloads_memberships):
            page_membership_arr = tmem_s.load_next_page_memberships(
                page_membership_arr=page_membership_arr
            )
        if cutlass.const_expr(releases_sage_stage_after_max):
            # The scores came back dequantized; the P pass reads no words.
            sage_k_scales.release()
        if cutlass.const_expr(
            (not cfg.use_keeps_mma_ab or not cfg.uses_tmem_p)
            and not cfg.streams_scores_in_q32_bands
        ):
            # ConsRelease: free S once the scores are in registers unless
            # a Keeps TMEM-P operand or streamed Swaps P pass still reads them.
            tmem_s.release()
        # Publish old/new max before the P path so correction can observe
        # the same stats order as the decode pipeline.
        # ProdWork: store old/new max for correction's in-loop O update.
        tmem_softmax_local.acquire()
        tmem_softmax_local.store_loop_old_new_stats(
            old_max_arr=old_max_arr,
            new_max_arr=new_max_arr,
            sum_arr=sum_arr,
        )
        tmem_softmax_local.commit()
        if cutlass.const_expr(cfg.streams_tmem_p_fragments):
            # One rolled loop streams every K32 probability fragment; the
            # fragment body exists once in the instruction stream.
            p_fragments_kwargs = dict(new_max_arr=new_max_arr)
            if cutlass.const_expr(cfg.use_sage_attention):
                p_fragments_kwargs["sage_q_scale"] = sage_q_scale
                p_fragments_kwargs["sage_k_stage_bytes"] = sage_k_stage_bytes
            if cutlass.const_expr(cfg.use_block_sparse_proxy_routes):
                p_fragments_kwargs["route_flags"] = sparse_route_flags
            compute_p_fragments(**p_fragments_kwargs)
        else:
            # Wait for a free P stage before entering the ordered window so
            # BMM2 backpressure on this group's P pipeline cannot extend the
            # baton hold and stall the partner softmax group.
            smem_p.acquire()
            if tmem_softmax_order is not None:
                if inst_id == 0:
                    tmem_softmax_order.wait_softmax0()
                else:
                    tmem_softmax_order.wait_softmax1()
            # ProdWork: compute P=exp(S-new_max), store it in the profile's
            # SMEM or staged-TMEM operand, and record local sums for the
            # running softmax sum update.
            if cutlass.const_expr(cfg.use_block_sparse_proxy_routes):
                smem_p.compute_proxy_route_p(
                    new_max_arr=new_max_arr,
                    s_arr=s_arr,
                    keeps_route_flags=sparse_route_flags,
                    swaps_route_flags=sparse_token_word2,
                )
            elif cutlass.const_expr(use_sage):
                smem_p.compute_sage_p(
                    new_max_arr=new_max_arr,
                    s_arr=s_arr,
                    sage_q_scale=sage_q_scale,
                )
            else:
                smem_p.compute_p(
                    new_max_arr=new_max_arr,
                    s_arr=s_arr,
                )  # publishes the local denominator through tmem_s
            smem_p.commit()
            if tmem_softmax_order is not None:
                if inst_id == 0:
                    tmem_softmax_order.release_softmax1()
                else:
                    tmem_softmax_order.release_softmax0()
        if cutlass.const_expr(
            (cfg.use_keeps_mma_ab and cfg.uses_tmem_p)
            or cfg.streams_scores_in_q32_bands
        ):
            # P has consumed the retained S columns, so the next QK wave
            # can now overwrite them.
            tmem_s.release()
        if cutlass.const_expr(use_sage and not releases_sage_stage_after_max):
            # ConsRelease: both passes have read this tile's ``sfK`` words.
            sage_k_scales.release()
        # ProdWork: FP8 path applies the cross-resource sum correction
        # before TmemS.reduce_sums publishes the new running sums.
        tmem_softmax_global.global_correction(
            old_max_arr=old_max_arr,
            new_max_arr=new_max_arr,
            sum_arr=sum_arr,
        )  # publishes the corrected denominator through tmem_s
        # ConsTailWork: update the running sum after P is available.
        sum_arr = tmem_s.reduce_sums(
            old_max_arr=old_max_arr,
            new_max_arr=new_max_arr,
            sum_arr=sum_arr,
        )
        if cutlass.const_expr(not cfg.use_keeps_mma_ab):
            with d.last_iter():
                # LastIter ProdWork: store the final sums for correction's
                # normalization/output tail.
                tmem_softmax_local.acquire()
                tmem_softmax_local.store_loop_sum_new_stats(
                    old_max_arr=old_max_arr,
                    new_max_arr=new_max_arr,
                    sum_arr=sum_arr,
                )
                tmem_softmax_local.commit()
    if cutlass.const_expr(cfg.use_keeps_mma_ab):
        # The final sum payload has no matching QK/StatsDone token.
        tmem_softmax_local.acquire()
        tmem_softmax_local.store_tail_sum_new_stats(
            old_max_arr=old_max_arr,
            new_max_arr=new_max_arr,
            sum_arr=sum_arr,
        )
        tmem_softmax_local.commit()
        if cutlass.const_expr(
            inst_id == 0
            and cfg.use_persistent_scheduler
            and cfg.uses_staged_one_inst_tmem_p
        ):
            # Tail stats add one stage beyond the S cadence. Advance the
            # second stats slot so both pipelines start the next work tile
            # on the same physical TMEM stage. A one-instance profile has
            # only the first instance.
            tmem_softmax_local.acquire()
            tmem_softmax_local.commit()
    if membership_lifetime is not None:
        membership_lifetime.release()
