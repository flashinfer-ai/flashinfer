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

"""TMEM P readiness between softmax and the PV MMA."""

import cutlass.cute as cute
import cutlass
from cutlass import Int32
from cutlass.cutlass_dsl import Constexpr
from cutlass.experimental.task_scheduling.memory import TmemAllocation
from cutlass.experimental.task_scheduling.resources import (
    MemoryResource,
    PipelineConfig,
    StageInfo,
    TaskLocalVariable,
    consumer_work,
)
from dataclasses import dataclass, field
from ..config import FmhaConfig
from typing import Any, Optional


# ---------------------------------------------------------------------------
# TmemPResource -- P-ready handoff from softmax to UMMA
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class TmemPResource(MemoryResource):
    """Pipeline-only P handoff for split S/P scheduling.

    Softmax stores P into the TMEM columns owned by ``TmemSPResource`` and
    commits this AsyncUmma resource. The MMA task waits on it before issuing
    PV, so the next QK can use the other S/P stage without using the S acquire
    as an implicit P-ready wait.
    """

    cfg: Constexpr[FmhaConfig] = field(init=False, default=None)
    tmem_p_base: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    tmem_p_offset: Constexpr[int] = field(init=False, default=None)

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        cfg: FmhaConfig,
        tmem_p_offset: int,
        **kwargs: Any,
    ) -> None:
        """Bind the base P TMEM offset used by the split S/P pipeline."""
        super().__init__(pipeline_config=pipeline_config, **kwargs)
        self.cfg = cfg
        self.tmem_p_offset = tmem_p_offset
        self.tmem_p_base = TaskLocalVariable(
            dtype=Int32,
            default=Int32(tmem_p_offset),
            docs="Selected staged P TMEM column base for PV MMA.",
        )

    @cute.jit
    def create_function_variables(self, context: Optional[Any] = None) -> Int32:
        """Create the staged P-base dataflow slot."""
        _ = context
        return Int32(self.tmem_p_offset)

    def get_tmem_requirements(self) -> list[TmemAllocation]:
        """Return no allocation because P aliases the TmemSP allocation."""
        return []

    @consumer_work(returns=("tmem_p_base",))
    @cute.jit
    def p_base(self, stage_info: StageInfo) -> Int32:
        """Return the staged P column base for the MMA PV producer."""
        tmem_p_base = Int32(self.tmem_p_offset)
        if cutlass.const_expr(self.cfg.mma_softmax_stage > 1):
            tmem_p_base = tmem_p_base + stage_info.stage_idx * self.cfg.qk_mma_tiler[1]
        return tmem_p_base
