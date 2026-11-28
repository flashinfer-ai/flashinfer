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

"""Softmax tasks of the context kernel."""

from collections.abc import Callable
from cutlass.experimental.task_scheduling.resources import WorkQueue
from cutlass.experimental.task_scheduling.schedule_builder import domain_loop, schedule
from cutlass.experimental.task_scheduling.task import Task
from dataclasses import dataclass
from .....device.stage import WorkStage
from ..resources.smem_p import SmemPResource
from ..resources.smem_sage_k_scales import SmemSageKScalesResource
from ..resources.sync import S0S1SequenceResource, TmemPPrefixReadyResource
from ..resources.tmem_p import TmemPResource
from ..resources.tmem_sp import TmemSPResource
from ..resources.tmem_stats import TmemStatsResource
from .common import (
    captured_loop_bounds,
    packed_context_skip_predicate,
    schedule_with_work_queue,
    src_resources,
    work_tile_schedule_loop,
)
from ....sage.modes import SageKTransport, SageScoreScaling
from typing import Any


@dataclass
class LoopMaskState:
    """Per-work-tile inputs of one softmax warp group's LOOP score mask."""

    q_offset: Any
    seqlen_k: Any = None
    window_start: Any = None
    window_end: Any = None


def cache_loop_mask_state(
    sp: TmemSPResource, tmem_sp: TmemSPResource, q_offset: Any
) -> LoopMaskState:
    """Cache a work tile's mask inputs and unmasked LOOP tile range (HEAD).

    Runs once per work tile, before the K/V loop. Each warp's unmasked range
    ``[begin, end)`` splits the LOOP into left-masked, unmasked, and
    right-masked tiles, so ``loop_row_max`` selects the masked path with one
    test. Window masks store the range in SMEM here; masks with a single right
    bound test it in key space from the cached ``q_offset`` / ``seqlen_k``,
    which their masked tiles reuse (``TmemSPResource.uses_loop_unmasked_tile_range``).
    """
    if tmem_sp.uses_varlen_q_offset_cache:
        q_offset = sp.cache_q_offset()
    mask = LoopMaskState(q_offset=q_offset)
    if tmem_sp.uses_packed_dense_k_mask:
        mask.seqlen_k = sp.cache_seqlen_k()
    if tmem_sp.uses_variable_window:
        mask.window_start, mask.window_end = sp.cache_variable_window_bounds()
    if tmem_sp.uses_loop_unmasked_tile_range:
        kind = tmem_sp.loop_mask_kind
        if kind == "variable_window":
            sp.cache_unmasked_tile_range_window(
                window_start=mask.window_start, window_end=mask.window_end
            )
        else:
            sp.cache_unmasked_tile_range_q_offset(q_offset=mask.q_offset)
    return mask


def create_softmax_task(
    index: int,
    tmem_sp: TmemSPResource,
    tmem_vec: TmemStatsResource,
    tmem_p: TmemPResource | None,
    s0s1_seq: S0S1SequenceResource | None,
    work_queue: WorkQueue | None,
    task_class: type[Task] = Task,
    tmem_p_prefix_ready: TmemPPrefixReadyResource | None = None,
    smem_p: SmemPResource | None = None,
    smem_sage_k: SmemSageKScalesResource | None = None,
    **task_kwargs: Any,
) -> Task:
    """Create a four-warp Softmax task.

    index=0: warps 0-3 (Softmax0Task) — S0-S1 producer (acquire/commit)
    index=1: warps 4-7 (Softmax1Task) — S0-S1 consumer (wait/release)

    Args:
        task_class: Task subclass used to instantiate the softmax schedule.
    """
    loop_start, loop_end, loop_step = captured_loop_bounds(task_class, task_kwargs)
    skip_work_tile_if = packed_context_skip_predicate(work_queue)

    # A missing S0-S1 sequence resource selects the single-QKV-instance path.
    # For D>128, the TMEM P pipeline gives S and P independent readiness
    # handoffs. MMA can issue next-tile QK on one stage while previous-tile PV
    # uses the other.
    if s0s1_seq is None:
        if tmem_p is not None and tmem_sp.cfg.has_tmem_p_pipeline:
            src = src_resources(tmem_sp, work_queue=work_queue)
            dst = [tmem_vec, tmem_p]

            @schedule
            def softmax_schedule(
                sp: TmemSPResource,
                vec: TmemStatsResource,
                tp: TmemPResource,
                wq: WorkQueue | None = None,
            ) -> None:
                p_chunk = sp.init_softmax_state()
                scale_softmax_log2 = sp.load_scale_softmax_log2()
                vec.init_store_state()
                with work_tile_schedule_loop(wq, skip_if=skip_work_tile_if):
                    old_row_max, row_max, row_sum, q_offset = (
                        sp.init_softmax_work_tile_state()
                    )
                    vec.init_store_work_tile_state()
                    mask = cache_loop_mask_state(sp, tmem_sp, q_offset)
                    q_offset = mask.q_offset
                    vec.acquire()
                    with domain_loop(loop_start, loop_end, loop_step):
                        # Softmax(i): wait for QK(Q,Ki) -> S(i).
                        sp.wait()
                        old_row_max, row_max = loop_row_max(
                            sp,
                            tmem_sp,
                            row_max,
                            mask,
                            lambda: sp.compute_row_max(row_max=row_max),
                        )
                        if tmem_sp.cfg.corr_skip_threshold_log2 > 0:
                            row_max = sp.freeze_row_max(
                                old_row_max=old_row_max,
                                row_max=row_max,
                                scale_softmax_log2=scale_softmax_log2,
                            )
                        # Stats(i): S(i) -> row max/sum for correction.
                        vec.store_vec(
                            old_row_max=old_row_max,
                            row_max=row_max,
                            row_sum=row_sum,
                        )
                        vec.commit()
                        # P(i): acquire the matching P-ready handoff stage.
                        tp.acquire()
                        # P(i): exp2(S(i)) -> P(i) in the same TMEM stage.
                        p_chunk = sp.exp2_p(
                            row_max=row_max,
                            scale_softmax_log2=scale_softmax_log2,
                        )
                        # P(i) ready: P(i) -> PV(Pi,Vi).
                        tp.commit()
                        # S/P(i): release softmax ownership for the next QK stage.
                        sp.release()
                        # Aux(i): finish the row-sum reduction after releasing SP.
                        row_sum = sp.softmax_aux_reduce(
                            old_row_max=old_row_max,
                            row_max=row_max,
                            row_sum=row_sum,
                            p_chunk=p_chunk,
                            scale_softmax_log2=scale_softmax_log2,
                        )
                        vec.acquire()

                    if tmem_sp.uses_head_paired_causal_tail_mask:
                        # Tail S: consume and mask the final head-paired score tile.
                        sp.wait()
                        old_row_max, row_max = sp.right_masked_row_max(
                            row_max=row_max,
                            q_offset=q_offset,
                            section=WorkStage.Tail,
                        )
                        vec.store_vec(
                            old_row_max=old_row_max,
                            row_max=row_max,
                            row_sum=row_sum,
                        )
                        vec.commit()
                        # Tail P: publish the final probability tile to PV.
                        tp.acquire()
                        p_chunk = sp.exp2_p(
                            row_max=row_max,
                            scale_softmax_log2=scale_softmax_log2,
                        )
                        tp.commit()
                        sp.release()
                        row_sum = sp.softmax_aux_reduce(
                            old_row_max=old_row_max,
                            row_max=row_max,
                            row_sum=row_sum,
                            p_chunk=p_chunk,
                            scale_softmax_log2=scale_softmax_log2,
                        )
                        vec.acquire()
                        # Drain the final SP/P-ready slots and publish identity stats.
                        sp.wait()
                        sp.release()
                        tp.acquire()
                        tp.commit()
                        old_row_max = sp.softmax_aux_identity(row_max=row_max)
                        vec.store_vec(
                            old_row_max=old_row_max,
                            row_max=row_max,
                            row_sum=row_sum,
                            final_stats=True,
                        )
                        vec.commit()
                    elif tmem_sp.uses_query_paired_causal_tail_mask:
                        # Tail S: consume and mask the final query-paired score tile.
                        sp.wait()
                        old_row_max, row_max = sp.masked_row_max(
                            row_max=row_max,
                            q_offset=q_offset,
                        )
                        vec.store_vec(
                            old_row_max=old_row_max,
                            row_max=row_max,
                            row_sum=row_sum,
                        )
                        vec.commit()
                        # Tail P: publish the final probability tile to PV.
                        tp.acquire()
                        p_chunk = sp.masked_exp2_p(
                            row_max=row_max,
                            scale_softmax_log2=scale_softmax_log2,
                        )
                        tp.commit()
                        sp.release()
                        row_sum = sp.softmax_aux_reduce(
                            old_row_max=old_row_max,
                            row_max=row_max,
                            row_sum=row_sum,
                            p_chunk=p_chunk,
                            scale_softmax_log2=scale_softmax_log2,
                        )
                        if tmem_sp.uses_query_paired_invalid_tail:
                            # Invalid peer tail: consume its padded SP slot without PV.
                            sp.wait()
                            old_row_max, row_max = sp.invalid_row_max(row_max=row_max)
                            vec.acquire()
                            vec.store_vec(
                                old_row_max=old_row_max,
                                row_max=row_max,
                                row_sum=row_sum,
                            )
                            vec.commit()
                            sp.invalid_exp2_p(row_max=row_max)
                            sp.release()
                        # Drain the final SP/P-ready slots and publish identity stats.
                        sp.wait()
                        sp.release()
                        tp.acquire()
                        tp.commit()
                        old_row_max = sp.softmax_aux_identity(row_max=row_max)
                        vec.acquire()
                        vec.store_vec(
                            old_row_max=old_row_max,
                            row_max=row_max,
                            row_sum=row_sum,
                            final_stats=True,
                        )
                        vec.commit()
                    elif tmem_sp.cfg.is_causal:
                        # Tail S: consume and causally mask the final score tile.
                        sp.wait()
                        old_row_max, row_max = sp.masked_row_max(
                            row_max=row_max,
                            q_offset=q_offset,
                        )
                        vec.store_vec(
                            old_row_max=old_row_max,
                            row_max=row_max,
                            row_sum=row_sum,
                        )
                        vec.commit()
                        # Tail P: publish the final probability tile to PV.
                        tp.acquire()
                        p_chunk = sp.masked_exp2_p(
                            row_max=row_max,
                            scale_softmax_log2=scale_softmax_log2,
                        )
                        tp.commit()
                        sp.release()
                        row_sum = sp.softmax_aux_reduce(
                            old_row_max=old_row_max,
                            row_max=row_max,
                            row_sum=row_sum,
                            p_chunk=p_chunk,
                            scale_softmax_log2=scale_softmax_log2,
                        )
                        # Drain the final SP/P-ready slots and publish identity stats.
                        sp.wait()
                        sp.release()
                        tp.acquire()
                        tp.commit()
                        old_row_max = sp.softmax_aux_identity(row_max=row_max)
                        vec.acquire()
                        vec.store_vec(
                            old_row_max=old_row_max,
                            row_max=row_max,
                            row_sum=row_sum,
                            final_stats=True,
                        )
                        vec.commit()
                    else:
                        # Non-causal cleanup: drain SP and the matching P-ready slot.
                        sp.wait()
                        sp.release()
                        tp.acquire()
                        tp.commit()
                        old_row_max = sp.softmax_aux_identity(row_max=row_max)
                        vec.store_vec(
                            old_row_max=old_row_max,
                            row_max=row_max,
                            row_sum=row_sum,
                            final_stats=True,
                        )
                        vec.commit()
                    if tmem_sp.cfg.stats_via_smem:
                        # Balance the two-stage stats cursor before the next
                        # captured persistent work tile.  The context task
                        # runtime carries pipeline state across work tiles,
                        # while each tile's static call layout begins at the
                        # same stage; the empty record keeps both in phase.
                        vec.acquire()
                        vec.store_vec(
                            old_row_max=old_row_max,
                            row_max=row_max,
                            row_sum=row_sum,
                        )
                        vec.commit()

            captured_schedule = schedule_with_work_queue(
                softmax_schedule, tmem_sp, tmem_vec, tmem_p, work_queue=work_queue
            )
            return task_class(
                src_resources=src,
                dst_resources=dst,
                warp_idx=index * 4,
                num_warps=4,
                schedule=captured_schedule,
                num_registers=tmem_sp.cfg.num_regs_softmax,
                name=f"Softmax{index}Task",
                **task_kwargs,
            )

        # Non-split single-instance fallback: softmax writes P into the current
        # SP stage and releases that same resource for MMA to consume directly.
        src = src_resources(tmem_sp, work_queue=work_queue)
        dst = [tmem_vec]

        @schedule
        def softmax_schedule(
            sp: TmemSPResource,
            vec: TmemStatsResource,
            wq: WorkQueue | None = None,
        ) -> None:
            old_row_max, row_max, row_sum, p_chunk, q_offset = (
                sp.create_function_variables()
            )
            vec.create_function_variables()
            with work_tile_schedule_loop(wq, skip_if=skip_work_tile_if):
                if wq is not None:
                    old_row_max, row_max, row_sum, q_offset = (
                        sp.create_work_tile_variables(
                            old_row_max=old_row_max,
                            row_max=row_max,
                            row_sum=row_sum,
                            q_offset=q_offset,
                        )
                    )
                    vec.create_work_tile_variables()
                mask = cache_loop_mask_state(sp, tmem_sp, q_offset)
                q_offset = mask.q_offset
                vec.acquire()
                with domain_loop(loop_start, loop_end, loop_step):
                    sp.wait()
                    old_row_max, row_max = loop_row_max(
                        sp,
                        tmem_sp,
                        row_max,
                        mask,
                        lambda: sp.row_max(row_max),
                    )
                    vec.store_vec(
                        old_row_max,
                        row_max,
                        row_sum,
                    )
                    vec.commit()
                    p_chunk = sp.exp2_p(row_max)
                    sp.release()
                    row_sum = sp.softmax_post_release_reduce(
                        old_row_max, row_max, row_sum, p_chunk
                    )
                    vec.acquire()

                if tmem_sp.uses_head_paired_causal_tail_mask:
                    sp.wait()
                    old_row_max, row_max = sp.right_masked_row_max(
                        row_max=row_max,
                        q_offset=q_offset,
                        section=WorkStage.Tail,
                    )
                    vec.store_vec(
                        old_row_max,
                        row_max,
                        row_sum,
                    )
                    vec.commit()
                    p_chunk = sp.exp2_p(row_max)
                    sp.release()
                    row_sum = sp.softmax_post_release_reduce(
                        old_row_max, row_max, row_sum, p_chunk
                    )
                    vec.acquire()
                    sp.wait()
                    sp.release()
                    old_row_max = sp.softmax_post_release_identity(row_max)
                    vec.store_vec(
                        old_row_max,
                        row_max,
                        row_sum,
                        final_stats=True,
                    )
                    vec.commit()
                elif tmem_sp.uses_query_paired_causal_tail_mask:
                    sp.wait()
                    old_row_max, row_max = sp.masked_row_max(
                        row_max=row_max,
                        q_offset=q_offset,
                    )
                    vec.store_vec(old_row_max, row_max, row_sum)
                    vec.commit()
                    p_chunk = sp.masked_exp2_p(
                        row_max=row_max,
                    )
                    sp.release()
                    row_sum = sp.softmax_post_release_reduce(
                        old_row_max, row_max, row_sum, p_chunk
                    )
                    if tmem_sp.uses_query_paired_invalid_tail:
                        sp.wait()
                        old_row_max, row_max = sp.invalid_row_max(row_max)
                        vec.acquire()
                        vec.store_vec(old_row_max, row_max, row_sum)
                        vec.commit()
                        sp.invalid_exp2_p(row_max=row_max)
                        sp.release()
                    sp.wait()
                    sp.release()
                    old_row_max = sp.softmax_post_release_identity(row_max)
                    vec.acquire()
                    vec.store_vec(
                        old_row_max,
                        row_max,
                        row_sum,
                        final_stats=True,
                    )
                    vec.commit()
                elif tmem_sp.cfg.is_causal:
                    sp.wait()
                    old_row_max, row_max = sp.masked_row_max(
                        row_max=row_max,
                        q_offset=q_offset,
                    )
                    vec.store_vec(old_row_max, row_max, row_sum)
                    vec.commit()
                    p_chunk = sp.masked_exp2_p(
                        row_max=row_max,
                    )
                    sp.release()
                    row_sum = sp.softmax_post_release_reduce(
                        old_row_max, row_max, row_sum, p_chunk
                    )
                    sp.wait()
                    sp.release()
                    old_row_max = sp.softmax_post_release_identity(row_max)
                    vec.acquire()
                    vec.store_vec(
                        old_row_max,
                        row_max,
                        row_sum,
                        final_stats=True,
                    )
                    vec.commit()
                else:
                    sp.wait()
                    sp.release()
                    old_row_max = sp.softmax_post_release_identity(row_max)
                    vec.store_vec(old_row_max, row_max, row_sum)
                    vec.commit()

        captured_schedule = schedule_with_work_queue(
            softmax_schedule, tmem_sp, tmem_vec, work_queue=work_queue
        )
        return task_class(
            src_resources=src,
            dst_resources=dst,
            warp_idx=index * 4,
            num_warps=4,
            schedule=captured_schedule,
            num_registers=tmem_sp.cfg.num_regs_softmax,
            name=f"Softmax{index}Task",
            **task_kwargs,
        )

    # Paired QKV instances use separate SP resources. P reaches the MMA through
    # the SP release, the P-prefix barrier (first half), TmemP or SmemP.
    # The sequence token paces peer progress; FP8 overlaps P computation.
    if s0s1_seq is not None and index == 1:
        src = src_resources(tmem_sp, s0s1_seq, work_queue=work_queue)
    else:
        src = src_resources(tmem_sp, work_queue=work_queue)
    dst = [tmem_vec]
    if s0s1_seq is not None and index == 0:
        dst.append(s0s1_seq)
    pv_half_overlap = tmem_sp.cfg.pv_half_overlap
    if pv_half_overlap:
        if tmem_p_prefix_ready is None:
            raise ValueError("pv_half_overlap requires the P-prefix barrier")
        dst.append(tmem_p_prefix_ready)
    p_in_smem = tmem_sp.cfg.p_in_smem
    if p_in_smem:
        if smem_p is None:
            raise ValueError("p_in_smem requires the group's SMEM P resource")
        dst.append(smem_p)
    vc_attention = tmem_sp.cfg.vc_attention
    # VC-Attention V treatment: tile means restored in-kernel, or V repair tiles.
    vc_restores_means = tmem_sp.cfg.vc_restores_means
    vc_repairs_v = vc_attention and not vc_restores_means
    # Sage attention takes each K tile's sfK words from the prepared buffer
    # or from the load warp's ring (``SageKTransport``), and scales blocks of
    # raw scores or dequantizes them (``SageScoreScaling``).
    use_sage = tmem_sp.cfg.use_sage_attention
    sage_transport = tmem_sp.cfg.sage_k_transport
    sage_scaling = tmem_sp.cfg.sage_score_scaling
    if use_sage and not p_in_smem:
        raise ValueError("Sage attention requires P in SMEM")
    if sage_transport is SageKTransport.SMEM and sage_scaling is SageScoreScaling.BLOCK:
        # The P pass of block-scaled scores reuses the max pass's words, and
        # a ring stage is released after the max pass.
        raise ValueError("block-scaled Sage scores load their sfK words directly")
    if sage_transport is SageKTransport.SMEM:
        if smem_sage_k is None:
            raise ValueError("this Sage K block size requires the sfK ring")
        src.append(smem_sage_k)

    def softmax_schedule_body(
        sp: TmemSPResource,
        vec: TmemStatsResource,
        seq: S0S1SequenceResource,
        wq: WorkQueue | None = None,
        pr: TmemPPrefixReadyResource | None = None,
        pb: SmemPResource | None = None,
        skr: SmemSageKScalesResource | None = None,
    ) -> None:
        """Captured schedule for one softmax warp group."""
        if tmem_sp.enable_early_tile_sum:
            # The contribution is produced and consumed inside each iteration;
            # do not carry even the scalar tile sum through the persistent loop.
            sp.init_softmax_state_early()
        else:
            p_chunk = sp.init_softmax_state()
        scale_softmax_log2 = sp.load_scale_softmax_log2()

        # VC-Attention sums the E4M3 probability codes inside the P store,
        # so vc_store_p returns the row sum and softmax_aux_reduce is skipped.
        vc_state: dict[str, Any] = {
            "prev_sum": None,
            "row_sum": None,
            "sums": None,
            "k_next": None,
            "kept": None,
        }

        def exp2_p(
            sp: TmemSPResource,
            *,
            row_max: Any,
            scale_softmax_log2: Any,
            old_row_max: Any = None,
            is_tail: bool = False,
            sage_q_scale: Any = None,
            sage_k_words: Any = None,
        ) -> Any:
            """Softmax and P store. With P in SMEM the store goes to the SMEM P tile,
            with the half overlap the leading half is published behind ``pr``, else P goes
            to the TMEM S/P stage. VC-Attention then carries the group row sums
            (rescaled to this tile's max) and stores the completed group's operand
            for the deferred mean step. A Sage plan passes the work tile's ``sfQ`` and, when it
            scales blocks, the tile's ``sfK`` words from its max pass."""
            if sage_scaling is SageScoreScaling.BLOCK:
                p_chunk = sp.sage_exp2_p_smem(
                    row_max=row_max,
                    scale_softmax_log2=scale_softmax_log2,
                    sage_q_scale=sage_q_scale,
                    sage_k_words=sage_k_words,
                )
                pb.acquire()
                sp.store_p()
                pb.commit()
                return p_chunk
            if use_sage:
                # The scores carry sfK already; sfQ joins the exp2 scale.
                p_chunk = sp.sage_dequantized_exp2_p_smem(
                    row_max=row_max,
                    scale_softmax_log2=scale_softmax_log2,
                    sage_q_scale=sage_q_scale,
                )
                pb.acquire()
                sp.store_p()
                pb.commit()
                return p_chunk
            if p_in_smem:
                # The S stage is released already, so exp2 and the P store are
                # auxiliary work on the loaded S and the SMEM P tile.
                p_chunk = sp.exp2_p_smem(
                    row_max=row_max, scale_softmax_log2=scale_softmax_log2
                )
                pb.acquire()
                if vc_repairs_v:
                    out = sp.vc_store_p_repair(
                        old_row_max=old_row_max,
                        row_max=row_max,
                        row_sum=vc_state["row_sum"],
                        vc_kept_row_sum=vc_state["kept"],
                        p_chunk=p_chunk,
                        is_tail=is_tail,
                    )
                    vc_state["kept"], vc_state["row_sum"] = out[0], out[1]
                elif vc_restores_means:
                    out = sp.vc_store_p(
                        old_row_max=old_row_max,
                        row_max=row_max,
                        row_sum=vc_state["row_sum"],
                        vc_prev_tile_sum=vc_state["prev_sum"],
                        p_chunk=p_chunk,
                        is_tail=is_tail,
                        vc_meansums=vc_state["sums"],
                    )
                    vc_state["prev_sum"], vc_state["row_sum"] = out[0], out[1]
                    vc_state["sums"] = out[2]
                else:
                    sp.store_p()
                pb.commit()
                return p_chunk
            if not pv_half_overlap:
                return sp.exp2_p(row_max=row_max, scale_softmax_log2=scale_softmax_log2)
            pr.acquire()
            p_lo = sp.exp2_p_lo(row_max=row_max, scale_softmax_log2=scale_softmax_log2)
            pr.commit()
            return sp.exp2_p_hi(
                row_max=row_max, scale_softmax_log2=scale_softmax_log2, p_lo=p_lo
            )

        vec.init_store_state()
        with work_tile_schedule_loop(wq, skip_if=skip_work_tile_if):
            # Recompute per-tile SP/Vec TMEM state.
            old_row_max, row_max, row_sum, q_offset = sp.init_softmax_work_tile_state()
            vec.init_store_work_tile_state()
            vc_state["row_sum"] = row_sum
            if vc_attention:
                init = sp.vc_init_row_scale()
                vc_row_scale, vc_state["prev_sum"] = init[0], init[1]
                vc_state["sums"] = init[2]
                vc_state["kept"] = init[3]
                vc_state["k_next"] = init[4]
            mask = cache_loop_mask_state(sp, tmem_sp, q_offset)
            q_offset = mask.q_offset
            sage_q_scale = sage_k_words = None
            if use_sage:
                sage_q_scale, sage_k_first_atom = sp.cache_sage_scales()

            def start_sage_k_words(section: WorkStage) -> Any:
                """Start obtaining the K tile's ``sfK`` words before the wait for its scores.

                The global transport loads the words, so the wait hides their
                latency; the ring transport waits for the load warp's stage and
                returns its offset, read after the score wait."""
                if sage_transport is SageKTransport.GMEM:
                    return sp.load_sage_k_words(
                        sage_k_first_atom=sage_k_first_atom, section=section
                    )
                skr.wait()
                return skr.open_stage()

            def sage_row_max(row_max: Any, k_words: Any, section: WorkStage) -> Any:
                """Row max of the dequantized scores.

                A ring stage's words are read after the score wait and the
                stage is released after the max pass (``SmemSageKScalesResource``)."""
                if sage_transport is SageKTransport.SMEM:
                    k_words = sp.read_staged_sage_k_words(sage_k_stage_words=k_words)
                if sage_scaling is SageScoreScaling.BLOCK:
                    maxima = sp.sage_row_max(
                        row_max=row_max,
                        sage_q_scale=sage_q_scale,
                        sage_k_words=k_words,
                        section=section,
                    )
                else:
                    maxima = sp.sage_dequantized_row_max(
                        row_max=row_max,
                        sage_q_scale=sage_q_scale,
                        sage_k_words=k_words,
                        section=section,
                    )
                if sage_transport is SageKTransport.SMEM:
                    skr.release()
                return maxima

            # Reserve a stats slot before the first softmax result is published.
            vec.acquire()
            with domain_loop(loop_start, loop_end, loop_step):
                if use_sage:
                    sage_k_words = start_sage_k_words(WorkStage.Loop)
                sp.wait()
                # Compute row max and publish vec.
                if use_sage:
                    old_row_max, row_max = sage_row_max(
                        row_max, sage_k_words, WorkStage.Loop
                    )
                elif vc_attention and tmem_sp.loop_mask_kind in (
                    None,
                    "fixed_dense_k_tail",
                ):
                    old_row_max, row_max, vc_state["k_next"] = sp.vc_compute_row_max(
                        row_max=row_max,
                        vc_row_scale=vc_row_scale,
                        vc_k_scale_next=vc_state["k_next"],
                    )
                else:
                    # compute_row_max masks a fixed dense K tail itself.
                    old_row_max, row_max = loop_row_max(
                        sp,
                        tmem_sp,
                        row_max,
                        mask,
                        lambda: sp.compute_row_max(row_max=row_max),
                        fixed_dense_k_tail_in_fallback=True,
                    )
                if tmem_sp.cfg.corr_skip_threshold_log2 > 0:
                    row_max = sp.freeze_row_max(
                        old_row_max=old_row_max,
                        row_max=row_max,
                        scale_softmax_log2=scale_softmax_log2,
                    )
                vec.store_vec(
                    old_row_max=old_row_max,
                    row_max=row_max,
                    row_sum=row_sum,
                )
                vec.commit()
                if p_in_smem:
                    # S is in registers. Free the stage so QK(i+1) overlaps exp2.
                    sp.release()
                if s0s1_seq is None:
                    pass
                elif index == 0:
                    # Softmax0 is the S0-S1 producer: acquire/commit sequence.
                    seq.acquire()
                else:
                    # Softmax1 is the S0-S1 consumer: wait/release sequence.
                    seq.wait()
                # FP8 returns the pacing token before P work. P still reaches
                # the MMA through the SP release (TMEM P) or SmemP (SMEM P).
                early_token = tmem_sp.cfg.uses_d128_fp8_softmax_cadence or (
                    p_in_smem and tmem_sp.cfg.fp8_psmem_early_token
                )
                if early_token:
                    if index == 0:
                        seq.commit()
                    else:
                        seq.release()
                # Apply softmax and write P.
                p_chunk = exp2_p(
                    sp,
                    row_max=row_max,
                    scale_softmax_log2=scale_softmax_log2,
                    old_row_max=old_row_max,
                    sage_q_scale=sage_q_scale,
                    sage_k_words=sage_k_words,
                )
                if s0s1_seq is None or early_token:
                    pass
                elif index == 0:
                    seq.commit()
                else:
                    seq.release()
                if not p_in_smem:
                    sp.release()
                # Reduction.
                if vc_attention:
                    row_sum = vc_state["row_sum"]
                else:
                    row_sum = sp.softmax_aux_reduce(
                        old_row_max=old_row_max,
                        row_max=row_max,
                        row_sum=row_sum,
                        p_chunk=p_chunk,
                        scale_softmax_log2=scale_softmax_log2,
                    )
                # Acquire vec for next iter.
                vec.acquire()

            def publish_final_stats(row_max: Any, row_sum: Any) -> None:
                """Store the last mean-group row sums or take the repair denominator, drain
                the final SP slot and publish the identity stats of the work tile."""
                if vc_restores_means:
                    # The last row sums ride a second P handoff to the tail step.
                    pb.acquire()
                    sp.vc_store_rowsum_final(
                        vc_prev_tile_sum=vc_state["prev_sum"],
                        vc_meansums=vc_state["sums"],
                    )
                    pb.commit()
                elif vc_repairs_v:
                    # The repair tiles stay out of the denominator.
                    row_sum = vc_state["kept"]
                if not p_in_smem:
                    sp.wait()
                    sp.release()
                old_row_max = sp.softmax_aux_identity(row_max=row_max)
                vec.store_vec(
                    old_row_max=old_row_max,
                    row_max=row_max,
                    row_sum=row_sum,
                    final_stats=True,
                )
                vec.commit()

            if tmem_sp.uses_head_paired_causal_tail_mask:
                # Head-paired maps Q0/Q1 to adjacent Hq slices at the same S
                # tile. Its tail mask uses right_masked_row_max(), which does
                # not add the query-paired q_half * q_tile_m sequence advance.
                sp.wait()
                old_row_max, row_max = sp.right_masked_row_max(
                    row_max=row_max,
                    q_offset=q_offset,
                    section=WorkStage.Tail,
                )
                vec.store_vec(
                    old_row_max=old_row_max,
                    row_max=row_max,
                    row_sum=row_sum,
                )
                vec.commit()
                if s0s1_seq is None:
                    pass
                elif index == 0:
                    seq.acquire()
                else:
                    seq.wait()
                p_chunk = sp.exp2_p(
                    row_max=row_max,
                    scale_softmax_log2=scale_softmax_log2,
                )
                if s0s1_seq is None:
                    pass
                elif index == 0:
                    seq.commit()
                else:
                    seq.release()
                sp.release()
                if vc_attention:
                    row_sum = vc_state["row_sum"]
                else:
                    row_sum = sp.softmax_aux_reduce(
                        old_row_max=old_row_max,
                        row_max=row_max,
                        row_sum=row_sum,
                        p_chunk=p_chunk,
                        scale_softmax_log2=scale_softmax_log2,
                    )
                vec.acquire()
                sp.wait()
                sp.release()
                old_row_max = sp.softmax_aux_identity(row_max=row_max)
                vec.store_vec(
                    old_row_max=old_row_max,
                    row_max=row_max,
                    row_sum=row_sum,
                    final_stats=True,
                )
                vec.commit()
            elif tmem_sp.uses_query_paired_causal_tail_mask:
                # Query-paired maps Q1 to the next S tile. Its generic causal
                # tail uses masked_row_max(), which includes q_half * q_tile_m
                # so each peer tile is masked at the right sequence boundary.
                sp.wait()
                old_row_max, row_max = sp.masked_row_max(
                    row_max=row_max,
                    q_offset=q_offset,
                )
                vec.store_vec(
                    old_row_max=old_row_max,
                    row_max=row_max,
                    row_sum=row_sum,
                )
                vec.commit()
                if p_in_smem:
                    sp.release()
                if s0s1_seq is not None:
                    seq.acquire()
                if p_in_smem:
                    p_chunk = sp.masked_exp2_p_smem(
                        row_max=row_max,
                        scale_softmax_log2=scale_softmax_log2,
                    )
                    pb.acquire()
                    sp.store_p()
                    pb.commit()
                else:
                    p_chunk = sp.masked_exp2_p(
                        row_max=row_max,
                        scale_softmax_log2=scale_softmax_log2,
                    )
                if s0s1_seq is not None:
                    seq.commit()
                if not p_in_smem:
                    sp.release()
                if vc_attention:
                    row_sum = vc_state["row_sum"]
                else:
                    row_sum = sp.softmax_aux_reduce(
                        old_row_max=old_row_max,
                        row_max=row_max,
                        row_sum=row_sum,
                        p_chunk=p_chunk,
                        scale_softmax_log2=scale_softmax_log2,
                    )
                if tmem_sp.uses_query_paired_invalid_tail:
                    sp.wait()
                    old_row_max, row_max = sp.invalid_row_max(row_max=row_max)
                    vec.acquire()
                    vec.store_vec(
                        old_row_max=old_row_max,
                        row_max=row_max,
                        row_sum=row_sum,
                    )
                    vec.commit()
                    if s0s1_seq is not None:
                        seq.acquire()
                    sp.invalid_exp2_p(row_max=row_max)
                    if p_in_smem:
                        # MMA skips this PV but still waits on the P tile.
                        pb.acquire()
                        pb.commit()
                    if s0s1_seq is not None:
                        seq.commit()
                    sp.release()
                if not p_in_smem:
                    sp.wait()
                    sp.release()
                old_row_max = sp.softmax_aux_identity(row_max=row_max)
                vec.acquire()
                vec.store_vec(
                    old_row_max=old_row_max,
                    row_max=row_max,
                    row_sum=row_sum,
                    final_stats=True,
                )
                vec.commit()
            elif tmem_sp.cfg.is_causal:
                # Causal tail: mask the last tile, then publish the final stats.
                sp.wait()
                old_row_max, row_max = sp.masked_row_max(
                    row_max=row_max,
                    q_offset=q_offset,
                )
                vec.store_vec(
                    old_row_max=old_row_max,
                    row_max=row_max,
                    row_sum=row_sum,
                )
                vec.commit()
                if p_in_smem:
                    sp.release()
                if s0s1_seq is not None:
                    seq.wait()
                if p_in_smem:
                    p_chunk = sp.masked_exp2_p_smem(
                        row_max=row_max,
                        scale_softmax_log2=scale_softmax_log2,
                    )
                    pb.acquire()
                    sp.store_p()
                    pb.commit()
                else:
                    p_chunk = sp.masked_exp2_p(
                        row_max=row_max,
                        scale_softmax_log2=scale_softmax_log2,
                    )
                if s0s1_seq is not None:
                    seq.release()
                if not p_in_smem:
                    sp.release()
                if vc_attention:
                    row_sum = vc_state["row_sum"]
                else:
                    row_sum = sp.softmax_aux_reduce(
                        old_row_max=old_row_max,
                        row_max=row_max,
                        row_sum=row_sum,
                        p_chunk=p_chunk,
                        scale_softmax_log2=scale_softmax_log2,
                    )
                if not p_in_smem:
                    sp.wait()
                    sp.release()
                old_row_max = sp.softmax_aux_identity(row_max=row_max)
                vec.acquire()
                vec.store_vec(
                    old_row_max=old_row_max,
                    row_max=row_max,
                    row_sum=row_sum,
                    final_stats=True,
                )
                vec.commit()
            elif tmem_sp.uses_fixed_dense_k_tail_mask:
                # Dense tail: one more loop step with the zero-filled lanes masked out.
                if use_sage:
                    sage_k_words = start_sage_k_words(WorkStage.Tail)
                sp.wait()
                if vc_attention:
                    old_row_max, row_max = sp.vc_fixed_dense_k_tail_masked_row_max(
                        row_max=row_max,
                        vc_row_scale=vc_row_scale,
                        vc_k_scale_next=vc_state["k_next"],
                    )
                elif use_sage:
                    old_row_max, row_max = sage_row_max(
                        row_max, sage_k_words, WorkStage.Tail
                    )
                else:
                    old_row_max, row_max = sp.fixed_dense_k_tail_masked_row_max(
                        row_max=row_max,
                        section=WorkStage.Tail,
                    )
                vec.store_vec(
                    old_row_max=old_row_max,
                    row_max=row_max,
                    row_sum=row_sum,
                )
                vec.commit()
                if p_in_smem:
                    sp.release()
                if s0s1_seq is None:
                    pass
                elif index == 0:
                    seq.acquire()
                else:
                    seq.wait()
                p_chunk = exp2_p(
                    sp,
                    row_max=row_max,
                    scale_softmax_log2=scale_softmax_log2,
                    old_row_max=old_row_max,
                    is_tail=True,
                    sage_q_scale=sage_q_scale,
                    sage_k_words=sage_k_words,
                )
                if s0s1_seq is None:
                    pass
                elif index == 0:
                    seq.commit()
                else:
                    seq.release()
                if not p_in_smem:
                    sp.release()
                if vc_attention:
                    row_sum = vc_state["row_sum"]
                else:
                    row_sum = sp.softmax_aux_reduce(
                        old_row_max=old_row_max,
                        row_max=row_max,
                        row_sum=row_sum,
                        p_chunk=p_chunk,
                        scale_softmax_log2=scale_softmax_log2,
                    )
                vec.acquire()
                publish_final_stats(row_max, row_sum)
            else:
                # Non-causal tail: no more tiles, just publish the final stats.
                publish_final_stats(row_max, row_sum)

    @schedule
    def softmax_schedule(
        sp: TmemSPResource,
        vec: TmemStatsResource,
        seq: S0S1SequenceResource,
        wq: WorkQueue | None = None,
    ) -> None:
        softmax_schedule_body(sp, vec, seq, wq)

    @schedule
    def softmax_schedule_pr(
        sp: TmemSPResource,
        vec: TmemStatsResource,
        seq: S0S1SequenceResource,
        pr: TmemPPrefixReadyResource,
        wq: WorkQueue | None = None,
    ) -> None:
        softmax_schedule_body(sp, vec, seq, wq, pr)

    @schedule
    def softmax_schedule_pb(
        sp: TmemSPResource,
        vec: TmemStatsResource,
        seq: S0S1SequenceResource,
        pb: SmemPResource,
        wq: WorkQueue | None = None,
    ) -> None:
        softmax_schedule_body(sp, vec, seq, wq, None, pb)

    @schedule
    def softmax_schedule_pb_skr(
        sp: TmemSPResource,
        vec: TmemStatsResource,
        seq: S0S1SequenceResource,
        pb: SmemPResource,
        skr: SmemSageKScalesResource,
        wq: WorkQueue | None = None,
    ) -> None:
        softmax_schedule_body(sp, vec, seq, wq, None, pb, skr)

    if p_in_smem and sage_transport is SageKTransport.SMEM:
        captured_schedule = schedule_with_work_queue(
            softmax_schedule_pb_skr,
            tmem_sp,
            tmem_vec,
            s0s1_seq,
            smem_p,
            smem_sage_k,
            work_queue=work_queue,
        )
    elif p_in_smem:
        captured_schedule = schedule_with_work_queue(
            softmax_schedule_pb,
            tmem_sp,
            tmem_vec,
            s0s1_seq,
            smem_p,
            work_queue=work_queue,
        )
    elif pv_half_overlap:
        captured_schedule = schedule_with_work_queue(
            softmax_schedule_pr,
            tmem_sp,
            tmem_vec,
            s0s1_seq,
            tmem_p_prefix_ready,
            work_queue=work_queue,
        )
    else:
        captured_schedule = schedule_with_work_queue(
            softmax_schedule, tmem_sp, tmem_vec, s0s1_seq, work_queue=work_queue
        )
    return task_class(
        src_resources=src,
        dst_resources=dst,
        warp_idx=index * 4,
        num_warps=4,
        schedule=captured_schedule,
        num_registers=tmem_sp.cfg.num_regs_softmax,
        name=f"Softmax{index}Task",
        **task_kwargs,
    )


def loop_row_max(
    sp: TmemSPResource,
    tmem_sp: TmemSPResource,
    row_max: Any,
    mask: LoopMaskState,
    unmasked_row_max: Callable[[], Any],
    *,
    fixed_dense_k_tail_in_fallback: bool = False,
) -> Any:
    """Return ``(old_row_max, row_max)`` of one LOOP tile for every schedule.

    Dispatches on ``loop_mask_kind``; ``unmasked_row_max`` runs when the loop
    has no mask (and, with ``fixed_dense_k_tail_in_fallback``, when that
    fallback masks the fixed dense K tail itself).
    """
    kind = tmem_sp.loop_mask_kind
    if kind == "variable_window":
        return sp.variable_window_row_max(
            row_max=row_max,
            window_start=mask.window_start,
            window_end=mask.window_end,
        )
    if kind == "left_window":
        return sp.left_masked_row_max(row_max=row_max, q_offset=mask.q_offset)
    if kind == "varlen_right":
        return sp.right_masked_row_max(
            row_max=row_max, q_offset=mask.q_offset, section=WorkStage.Loop
        )
    if kind == "query_paired_causal":
        return sp.loop_masked_row_max(row_max=row_max, q_offset=mask.q_offset)
    if kind == "fixed_dense_k_tail" and not fixed_dense_k_tail_in_fallback:
        return sp.fixed_dense_k_tail_masked_row_max(row_max=row_max)
    if kind == "packed_dense_k":
        return sp.packed_dense_k_masked_row_max(
            row_max=row_max,
            seqlen_k=mask.seqlen_k,
            section=WorkStage.Loop,
        )
    return unmasked_row_max()
