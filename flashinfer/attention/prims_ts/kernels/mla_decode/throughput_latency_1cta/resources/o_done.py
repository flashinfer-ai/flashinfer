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

"""PV-completion resource of the column-split native FP4 QK schedule.

The 8-warp softmax task owns both 64-column halves of every 128-key tile and
rescales O itself (no correction task).  ``ODoneResource`` (UMMA -> Async, one
stage): the MMA issues the PV MMAs of tile k through its producer works and
commits after them ("PV(k) complete").  The softmax waits on it after
materializing P(k + 1) and before committing it, and rescales O by
exp2(c (m_k - m_{k+1})) when the max moved, so the rescale sits between PV(k)
and PV(k + 1) and the exp work of P(k + 1) overlaps PV(k).
"""

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Int32
from cutlass.experimental.task_scheduling.resources import (
    StageInfo,
    consumer_work,
    producer_work,
)

from ...helpers.ops import (
    freeze_smem_descriptor,
)
from .common import MlaResource

# =====================================================================
# ODoneResource — "PV(k) complete", carries the PV MMAs
# =====================================================================


@dataclass(kw_only=True)
class ODoneResource(MlaResource):
    """UMMA -> Async, one stage.

    Producer side (MMA warp): the PV MMAs of every tile are issued through
    ``pv_mma_loop`` / ``pv_mma_tail`` (P from the TMEM P stage, O in
    ``o_tmem_ref``) and ``commit`` tracks their completion.
    Consumer side (softmax task): ``rescale_o`` after the wait applies the
    lazy O rescale of its tile through the correction resource's TMEM O
    rescale (``corr_ref._rescale_loop_o``; it only needs the O column base,
    the softmax scale and the task cache).

    Not a subclass of ``TmemCorrResource``: under the DSL frontend a zero-arg
    ``super()`` inside a ``@cute.jit`` method resolves against the runtime
    type, so a subclass instance recursed in ``initialize_runtime_state_internal``.
    """

    o_tmem_ref: object = None
    corr_ref: object = None

    @cute.jit
    def _issue(
        self, stage_info: StageInfo, *, p_stage_idx, v_desc_0, scale_d, v_subtile_idx
    ):
        self.o_tmem_ref._issue_tmem_p_pv_mma(
            stage_info,
            p_stage_idx=p_stage_idx,
            v_desc=freeze_smem_descriptor(v_desc_0),
            scale_d=scale_d,
            v_subtile_idx=v_subtile_idx,
        )

    @producer_work
    @cute.jit
    def pv_mma_loop(
        self,
        stage_info: StageInfo,
        *,
        p_stage_idx,
        v_desc_0,
        v_subtile_idx: cutlass.Constexpr[int],
    ):
        """Loop PV of tile ``n + 1``: accumulate onto O (tile 0 initializes it)."""
        scale_d = stage_info.loop_offset != Int32(0)
        self._issue(
            stage_info,
            p_stage_idx=p_stage_idx,
            v_desc_0=v_desc_0,
            scale_d=scale_d,
            v_subtile_idx=v_subtile_idx,
        )

    @producer_work
    @cute.jit
    def pv_mma_tail(
        self,
        stage_info: StageInfo,
        *,
        p_stage_idx,
        v_desc_0,
        v_subtile_idx: cutlass.Constexpr[int],
    ):
        """Tail PV: accumulate unless no loop tile preceded it."""
        scale_d = self.o_tmem_ref._tail_has_prior_o(stage_info)
        self._issue(
            stage_info,
            p_stage_idx=p_stage_idx,
            v_desc_0=v_desc_0,
            scale_d=scale_d,
            v_subtile_idx=v_subtile_idx,
        )

    @consumer_work
    @cute.jit
    def pv_done_ack(self, stage_info: StageInfo) -> None:
        """Tail: consume the last PV's completion (no tile follows it)."""
        del stage_info

    @consumer_work
    @cute.jit
    def rescale_o(self, stage_info: StageInfo, *, old_max_arr, new_max_arr):
        """After PV(k) completed: O *= exp2(c (m_k - m_{k+1})) if the max moved.

        Identity lanes skip the exp2 and a whole warp of identity lanes skips
        the TMEM traffic (``_rescale_loop_o``); the ordering point before the
        P commit is kept either way.  Each 128-thread half of the softmax task
        rescales the two V stages of its own TMEM panel.
        """
        self.corr_ref._rescale_loop_o(
            stage_info,
            old_max_arr=old_max_arr,
            new_max_arr=new_max_arr,
            o_stage_idx=Int32(0),
        )
