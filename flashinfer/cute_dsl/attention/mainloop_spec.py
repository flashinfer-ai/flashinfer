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

from dataclasses import dataclass, replace
from typing import Dict, Union

from .config import AttentionConfig
from .mla_config import MLAConfig
from .tmem_layout import TmemLayout
from .warp_schedule import WarpSchedule, PREFILL_SCHEDULE
from .mla_warp_schedule import (
    MLAWarpSchedule,
    MLA_DECODE_SCHEDULE,
    MLAWarpScheduleFP8,
    MLA_DECODE_FP8_SCHEDULE,
)
from .pipeline_topology import (
    PipelineTopology,
    make_prefill_topology,
    make_prefill_topology_transform,
    make_mla_topology,
    make_mla_fp8_topology,
)


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

    has_logits_transform: bool = False

    q_stages: int = 2
    kv_stages: int = 3
    acc_stage: int = 1
    softmax_corr_stage: int = 1
    mma_corr_stage: int = 2
    mma_softmax_stage: int = 1
    epi_stage: int = 2
    buffer_align_bytes: int = 1024

    def resolve(self, dtype_width: int) -> MainloopSpec:
        """Return a new MainloopSpec with dtype-dependent stage counts resolved.

        Called after input dtype is known (inside __call__) but before
        SharedStorage or pipeline creation. The original spec is not modified.

        :param dtype_width: Bit width of the input element type.
        :returns: A new MainloopSpec with resolved kv_stages and pipeline_topology.
        """
        kv_stages = 4 if dtype_width == 8 else 3
        if self.has_logits_transform:
            topology = make_prefill_topology_transform(
                self.warp_schedule,
                q_stages=self.q_stages,
                kv_stages=kv_stages,
                mma_softmax_stages=self.mma_softmax_stage,
                epi_stages=self.epi_stage,
            )
        else:
            topology = make_prefill_topology(
                self.warp_schedule,
                q_stages=self.q_stages,
                kv_stages=kv_stages,
                mma_softmax_stages=self.mma_softmax_stage,
                softmax_corr_stages=self.softmax_corr_stage,
                mma_corr_stages=self.mma_corr_stage,
                epi_stages=self.epi_stage,
            )
        return replace(self, kv_stages=kv_stages, pipeline_topology=topology)

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
    has_logits_transform: bool = False,
) -> MainloopSpec:
    """Create a MainloopSpec for FMHA prefill.

    :param config: Core attention configuration.
    :param warp_schedule: Optional warp schedule override (defaults to PREFILL_SCHEDULE).
    :param has_logits_transform: If True, uses transform topology (no correction warp).
    """
    sched = warp_schedule if warp_schedule is not None else PREFILL_SCHEDULE
    tmem = TmemLayout.from_config(config)
    if has_logits_transform:
        topo = make_prefill_topology_transform(sched)
    else:
        topo = make_prefill_topology(sched)

    return MainloopSpec(
        config=config,
        warp_schedule=sched,
        tmem_layout=tmem,
        pipeline_topology=topo,
        has_logits_transform=has_logits_transform,
    )


# ---------------------------------------------------------------------------
#  MLA Decode
# ---------------------------------------------------------------------------


@dataclass
class MLAMainloopSpec:
    """Bundles pipeline topology and warp schedule for MLA decode kernels.

    Analogous to MainloopSpec but uses MLAConfig and MLAWarpSchedule.
    MLA stage counts are fixed (not dtype-dependent), so resolve() is a
    no-op that keeps the interface consistent with MainloopSpec.
    """

    config: MLAConfig
    warp_schedule: Union[MLAWarpSchedule, MLAWarpScheduleFP8]
    pipeline_topology: PipelineTopology

    buffer_align_bytes: int = 1024

    def resolve(self, dtype_width: int) -> MLAMainloopSpec:
        """No-op for MLA — stage counts are fixed. Returns self unchanged.

        Keeps the interface consistent with MainloopSpec so the kernel
        can call mainloop.resolve() uniformly.
        """
        return self

    def barrier_stage_counts(self) -> Dict[str, int]:
        """Return {edge_name: barrier_slot_count} for SharedStorage definition."""
        return {edge.name: edge.barrier_stages for edge in self.pipeline_topology.edges}


def make_mla_mainloop_spec(
    config: MLAConfig,
    warp_schedule: MLAWarpSchedule | None = None,
) -> MLAMainloopSpec:
    """Create an MLAMainloopSpec for MLA decode.

    :param config: MLA kernel configuration.
    :param warp_schedule: Optional warp schedule override (defaults to MLA_DECODE_SCHEDULE).
    """
    sched = warp_schedule if warp_schedule is not None else MLA_DECODE_SCHEDULE
    topo = make_mla_topology(
        sched,
        load_q_stages=config.load_q_stage,
        load_kv_stages=config.load_kv_stage,
        mma_s_stages=config.mma_s_stage,
        p_mma_stages=config.p_mma_stage,
        p_cor_stages=config.p_cor_stage,
        mma_o_stages=config.mma_o_stage,
        load_pt_stages=config.load_pt_stage,
        cluster_scale=config.cluster_shape_mnk[0],
    )
    return MLAMainloopSpec(
        config=config,
        warp_schedule=sched,
        pipeline_topology=topo,
    )


def make_mla_fp8_mainloop_spec(
    config: MLAConfig,
    warp_schedule: MLAWarpScheduleFP8 | None = None,
) -> MLAMainloopSpec:
    """Create an MLAMainloopSpec for FP8 MLA decode.

    Uses the FP8-specific pipeline topology with separate load_k/load_v
    pipelines and no load_pt pipeline.

    :param config: MLA kernel configuration (must have is_fp8=True).
    :param warp_schedule: Optional warp schedule override (defaults to MLA_DECODE_FP8_SCHEDULE).
    """
    sched = warp_schedule if warp_schedule is not None else MLA_DECODE_FP8_SCHEDULE
    topo = make_mla_fp8_topology(
        sched,
        load_q_stages=config.load_q_stage,
        load_k_stages=config.load_k_stage,
        load_v_stages=config.load_v_stage,
        mma_s_stages=config.mma_s_stage,
        p_mma_stages=config.p_mma_stage,
        p_cor_stages=config.p_cor_stage,
        mma_o_stages=config.mma_o_stage,
        cluster_scale=config.cluster_shape_mnk[0],
    )
    return MLAMainloopSpec(
        config=config,
        warp_schedule=sched,
        pipeline_topology=topo,
    )
