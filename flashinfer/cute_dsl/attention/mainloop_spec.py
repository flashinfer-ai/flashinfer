# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""MainloopSpec — the unit of composition for attention kernels.

Analogous to C++ CUTLASS's CollectiveMainloop (e.g.
Sm100FmhaFwdMainloopTmaWarpspecialized), this bundles:
- PipelineTopology (which pipelines connect which warps)
- TmemLayout (TMEM allocation map)
- WarpSchedule (warp role assignment and register budgets)
- Stage counts and buffer sizes
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .config import AttentionConfig
from .tmem_layout import TmemLayout
from .warp_schedule import WarpSchedule, PREFILL_SCHEDULE
from .pipeline_topology import PipelineTopology, make_prefill_topology


@dataclass
class MainloopSpec:
    """Bundles pipeline topology, TMEM layout, warp schedule, and stage counts.

    This is the Python-side equivalent of a C++ CUTLASS CollectiveMainloop
    template class. The kernel takes a MainloopSpec and creates all pipelines,
    tensors, and warp dispatch from it.

    Stage counts that depend on input dtype (e.g. kv_stages) are set by
    calling `resolve(dtype_width)` before use.
    """

    config: AttentionConfig
    warp_schedule: WarpSchedule
    tmem_layout: TmemLayout
    pipeline_topology: PipelineTopology

    q_stages: int = 2
    kv_stages: int = 3
    acc_stage: int = 1
    softmax_corr_stage: int = 1
    mma_corr_stage: int = 2
    mma_softmax_stage: int = 1
    epi_stage: int = 2
    buffer_align_bytes: int = 1024

    def resolve(self, dtype_width: int) -> None:
        """Resolve dtype-dependent stage counts.

        Called after input dtype is known (inside __call__) but before
        SharedStorage or pipeline creation.

        :param dtype_width: Bit width of the input element type.
        """
        self.kv_stages = 4 if dtype_width == 8 else 3
        self.pipeline_topology = make_prefill_topology(
            self.warp_schedule,
            q_stages=self.q_stages,
            kv_stages=self.kv_stages,
            mma_softmax_stages=self.mma_softmax_stage,
            softmax_corr_stages=self.softmax_corr_stage,
            mma_corr_stages=self.mma_corr_stage,
            epi_stages=self.epi_stage,
        )

    def barrier_stage_counts(self) -> Dict[str, int]:
        """Return {edge_name: barrier_slot_count} for SharedStorage definition.

        This is used to size the barrier storage arrays in the SharedStorage struct.
        """
        result = {}
        for edge in self.pipeline_topology.edges:
            result[edge.name] = edge.barrier_stages
        return result


def make_prefill_mainloop_spec(
    config: AttentionConfig,
    warp_schedule: WarpSchedule | None = None,
) -> MainloopSpec:
    """Create a MainloopSpec for FMHA prefill.

    :param config: Core attention configuration.
    :param warp_schedule: Optional warp schedule override (defaults to PREFILL_SCHEDULE).
    """
    sched = warp_schedule if warp_schedule is not None else PREFILL_SCHEDULE
    tmem = TmemLayout.from_config(config)
    topo = make_prefill_topology(sched)

    return MainloopSpec(
        config=config,
        warp_schedule=sched,
        tmem_layout=tmem,
        pipeline_topology=topo,
    )
