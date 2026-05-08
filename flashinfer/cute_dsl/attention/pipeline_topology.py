# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""PipelineTopology — declarative pipeline graph for attention kernels.

Replaces the imperative pipeline creation code (~76 lines of
make_pipeline_participants calls) with a declarative graph that can be
swapped between kernel variants (FMHA, decode).

Mirrors the C++ CUTLASS pattern where pipeline types are declared as
type aliases in the Mainloop collective, and the Kernel creates them.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any

import cutlass.pipeline as pipeline
from cutlass.pipeline import Agent, CooperativeGroup, PipelineProducer, PipelineConsumer

from .warp_schedule import WarpSchedule
from .mla_warp_schedule import MLAWarpSchedule, MLAWarpScheduleFP8


class PipelineType(enum.Enum):
    """Pipeline types available in the CuTe DSL pipeline library.

    Thread count rules per type:
    - TMA_UMMA: leader-only (len(warps)) for both producer and consumer
    - UMMA_ASYNC: leader-only for producer, all-threads for consumer
    - ASYNC_UMMA: all-threads for producer, leader-only for consumer (reverse of UMMA_ASYNC)
    - ASYNC: all-threads for both producer and consumer
    - CP_ASYNC: all-threads for both (cpasync-based, no cta_layout_vmnk/tx_count)
    """

    TMA_UMMA = "PipelineTmaUmma"
    UMMA_ASYNC = "PipelineUmmaAsync"
    ASYNC_UMMA = "PipelineAsyncUmma"
    ASYNC = "PipelineAsync"
    CP_ASYNC = "PipelineCpAsync"

    @property
    def cutlass_type(self):
        _map = {
            PipelineType.TMA_UMMA: pipeline.PipelineTmaUmma,
            PipelineType.UMMA_ASYNC: pipeline.PipelineUmmaAsync,
            PipelineType.ASYNC_UMMA: pipeline.PipelineAsyncUmma,
            PipelineType.ASYNC: pipeline.PipelineAsync,
            PipelineType.CP_ASYNC: pipeline.PipelineCpAsync,
        }
        return _map[self]

    @property
    def needs_cta_layout(self) -> bool:
        """Whether this pipeline type accepts cta_layout_vmnk for multi-CTA clusters."""
        return self in (
            PipelineType.TMA_UMMA,
            PipelineType.UMMA_ASYNC,
            PipelineType.ASYNC_UMMA,
        )

    def producer_thread_count(self, num_warps: int, threads_per_warp: int) -> int:
        if self in (PipelineType.TMA_UMMA, PipelineType.UMMA_ASYNC):
            return num_warps
        return threads_per_warp * num_warps

    def consumer_thread_count(self, num_warps: int, threads_per_warp: int) -> int:
        if self in (PipelineType.TMA_UMMA, PipelineType.ASYNC_UMMA):
            return num_warps
        return threads_per_warp * num_warps


@dataclass(frozen=True)
class PipelineEdge:
    """Describes a single pipeline in the topology.

    Each edge connects a producer role to a consumer role with a specific
    pipeline type and stage count.

    When cluster_scale > 1, the all-thread side of
    UMMA_ASYNC / ASYNC_UMMA pipelines multiplies its thread count by
    cluster_scale. TMA_UMMA pipelines are unaffected (leader-only on both sides).
    """

    name: str
    pipeline_type: PipelineType
    stages: int
    producer_warp_ids: Tuple[int, ...]
    consumer_warp_ids: Tuple[int, ...]
    tx_count_key: str | None = None
    cluster_scale: int = 1

    @property
    def barrier_field_name(self) -> str:
        return f"{self.name}_mbar_ptr"

    @property
    def barrier_stages(self) -> int:
        """Number of barrier slots needed. PipelineAsync uses 2x stages for the
        phase bit; others also use 2x for their internal bookkeeping."""
        if self.pipeline_type == PipelineType.ASYNC:
            return self.stages * 2
        return self.stages * 2


@dataclass
class PipelineTopology:
    """Declarative specification of the pipeline graph for an attention kernel.

    Contains all the PipelineEdge definitions. A factory method creates
    all pipeline participants from barrier storage and tx_count values.
    """

    edges: List[PipelineEdge] = field(default_factory=list)

    def edge_names(self) -> List[str]:
        return [e.name for e in self.edges]

    def get_edge(self, name: str) -> PipelineEdge:
        for e in self.edges:
            if e.name == name:
                return e
        raise KeyError(f"No pipeline edge named '{name}'")

    def create_pipelines(
        self,
        barrier_ptrs: Dict[str, Any],
        tx_counts: Dict[str, int],
        threads_per_warp: int,
        cta_layout_vmnk: Any = None,
    ) -> Dict[str, Tuple[PipelineProducer, PipelineConsumer]]:
        """Create all pipeline producer/consumer pairs from the topology.

        :param barrier_ptrs: Map from edge name to barrier storage pointer.
        :param tx_counts: Map from tx_count_key to byte count (for TMA pipelines).
        :param threads_per_warp: Threads per warp (typically 32).
        :param cta_layout_vmnk: CTA layout for multi-CTA clusters (None for single-CTA).
            Required by TMA_UMMA, UMMA_ASYNC, and ASYNC_UMMA when cluster_shape > 1.
        :returns: Dict mapping edge name to (producer, consumer) tuple.
        """
        result = {}
        for edge in self.edges:
            prod_threads = edge.pipeline_type.producer_thread_count(
                len(edge.producer_warp_ids), threads_per_warp
            )
            cons_threads = edge.pipeline_type.consumer_thread_count(
                len(edge.consumer_warp_ids), threads_per_warp
            )

            # Apply cluster_scale to the all-threads side of asymmetric pipelines
            if edge.cluster_scale > 1:
                pt = edge.pipeline_type
                if pt == PipelineType.UMMA_ASYNC:
                    cons_threads *= edge.cluster_scale
                elif pt == PipelineType.ASYNC_UMMA:
                    prod_threads *= edge.cluster_scale
                elif pt in (PipelineType.ASYNC, PipelineType.CP_ASYNC):
                    prod_threads *= edge.cluster_scale
                    cons_threads *= edge.cluster_scale

            create_kwargs = {
                "barrier_storage": barrier_ptrs[edge.name],
                "num_stages": edge.stages,
                "producer_group": CooperativeGroup(Agent.Thread, prod_threads),
                "consumer_group": CooperativeGroup(Agent.Thread, cons_threads),
                "defer_sync": True,
            }
            if edge.tx_count_key is not None:
                create_kwargs["tx_count"] = tx_counts[edge.tx_count_key]
            if cta_layout_vmnk is not None and edge.pipeline_type.needs_cta_layout:
                create_kwargs["cta_layout_vmnk"] = cta_layout_vmnk

            pipe = edge.pipeline_type.cutlass_type.create(**create_kwargs)
            result[edge.name] = pipe.make_participants()
        return result


def make_prefill_topology(
    schedule: WarpSchedule,
    q_stages: int = 2,
    kv_stages: int = 3,
    mma_softmax_stages: int = 1,
    softmax_corr_stages: int = 1,
    mma_corr_stages: int = 2,
    epi_stages: int = 2,
) -> PipelineTopology:
    """Build the pipeline topology for FMHA prefill.

    The prefill kernel has 9 pipelines connecting 6 warp roles::

        Load --[load_q]--> MMA --[mma_s0]--> Softmax0 --[s0_corr]--> Correction --[corr_epi]--> Epilogue
              --[load_kv]->     --[mma_s1]--> Softmax1 --[s1_corr]-->
                                --[mma_corr]------------------------------>
                                              Softmax0 --[s0_s1_seq]--> Softmax1

    :param schedule: Warp schedule defining warp role assignments.
    :param kv_stages: Stage count for KV pipeline (3 for fp16/bf16, 4 for fp8).
    """
    s = schedule
    load = (s.load_warp_id,)
    mma = (s.mma_warp_id,)
    s0 = s.softmax0_warp_ids
    s1 = s.softmax1_warp_ids
    corr = s.correction_warp_ids
    epi = (s.epilogue_warp_id,)

    return PipelineTopology(
        edges=[
            PipelineEdge(
                "load_q",
                PipelineType.TMA_UMMA,
                stages=q_stages,
                producer_warp_ids=load,
                consumer_warp_ids=mma,
                tx_count_key="q",
            ),
            PipelineEdge(
                "load_kv",
                PipelineType.TMA_UMMA,
                stages=kv_stages,
                producer_warp_ids=load,
                consumer_warp_ids=mma,
                tx_count_key="kv",
            ),
            PipelineEdge(
                "mma_s0",
                PipelineType.UMMA_ASYNC,
                stages=mma_softmax_stages,
                producer_warp_ids=mma,
                consumer_warp_ids=s0,
            ),
            PipelineEdge(
                "mma_s1",
                PipelineType.UMMA_ASYNC,
                stages=mma_softmax_stages,
                producer_warp_ids=mma,
                consumer_warp_ids=s1,
            ),
            PipelineEdge(
                "s0_corr",
                PipelineType.ASYNC,
                stages=softmax_corr_stages,
                producer_warp_ids=s0,
                consumer_warp_ids=corr,
            ),
            PipelineEdge(
                "s1_corr",
                PipelineType.ASYNC,
                stages=softmax_corr_stages,
                producer_warp_ids=s1,
                consumer_warp_ids=corr,
            ),
            PipelineEdge(
                "corr_epi",
                PipelineType.ASYNC,
                stages=epi_stages,
                producer_warp_ids=corr,
                consumer_warp_ids=epi,
            ),
            PipelineEdge(
                "mma_corr",
                PipelineType.UMMA_ASYNC,
                stages=mma_corr_stages,
                producer_warp_ids=mma,
                consumer_warp_ids=corr,
            ),
            # Softmax1 must wait for softmax0's row-max update before processing
            # its KV tile — online softmax requires sequential row-max propagation
            # between the two softmax warpgroups.
            PipelineEdge(
                "s0_s1_sequence",
                PipelineType.ASYNC,
                stages=1,
                producer_warp_ids=s0,
                consumer_warp_ids=s1,
            ),
        ]
    )


def make_prefill_topology_transform(
    schedule: WarpSchedule,
    q_stages: int = 2,
    kv_stages: int = 3,
    mma_softmax_stages: int = 1,
    epi_stages: int = 2,
) -> PipelineTopology:
    """Build the pipeline topology for FMHA prefill with logits_transform variants.

    No correction warp: softmax warps perform the epilog (TMEM->scale->SMEM)
    after their KV loop, then signal the epilogue warp directly.

    7 pipelines connecting 5 warp roles::

        Load --[load_q]--> MMA --[mma_s0]--> Softmax0 --[s0_epi]--> Epilogue
              --[load_kv]->     --[mma_s1]--> Softmax1 --[s1_epi]-->
                                Softmax0 --[s0_s1_seq]--> Softmax1

    :param schedule: Warp schedule defining warp role assignments.
    :param kv_stages: Stage count for KV pipeline (3 for fp16/bf16, 4 for fp8).
    """
    s = schedule
    load = (s.load_warp_id,)
    mma = (s.mma_warp_id,)
    s0 = s.softmax0_warp_ids
    s1 = s.softmax1_warp_ids
    epi = (s.epilogue_warp_id,)

    return PipelineTopology(
        edges=[
            PipelineEdge(
                "load_q",
                PipelineType.TMA_UMMA,
                stages=q_stages,
                producer_warp_ids=load,
                consumer_warp_ids=mma,
                tx_count_key="q",
            ),
            PipelineEdge(
                "load_kv",
                PipelineType.TMA_UMMA,
                stages=kv_stages,
                producer_warp_ids=load,
                consumer_warp_ids=mma,
                tx_count_key="kv",
            ),
            PipelineEdge(
                "mma_s0",
                PipelineType.UMMA_ASYNC,
                stages=mma_softmax_stages,
                producer_warp_ids=mma,
                consumer_warp_ids=s0,
            ),
            PipelineEdge(
                "mma_s1",
                PipelineType.UMMA_ASYNC,
                stages=mma_softmax_stages,
                producer_warp_ids=mma,
                consumer_warp_ids=s1,
            ),
            PipelineEdge(
                "s0_epi",
                PipelineType.ASYNC,
                stages=epi_stages,
                producer_warp_ids=s0,
                consumer_warp_ids=epi,
            ),
            PipelineEdge(
                "s1_epi",
                PipelineType.ASYNC,
                stages=epi_stages,
                producer_warp_ids=s1,
                consumer_warp_ids=epi,
            ),
            PipelineEdge(
                "s0_s1_sequence",
                PipelineType.ASYNC,
                stages=1,
                producer_warp_ids=s0,
                consumer_warp_ids=s1,
            ),
        ]
    )


def make_mla_topology(
    schedule: MLAWarpSchedule,
    load_q_stages: int = 1,
    load_kv_stages: int = 15,
    mma_s_stages: int = 2,
    p_mma_stages: int = 2,
    p_cor_stages: int = 2,
    mma_o_stages: int = 1,
    load_pt_stages: int = 4,
    cluster_scale: int = 2,
) -> PipelineTopology:
    """Build the pipeline topology for MLA decode.

    7 pipelines connecting 5 warp roles::

        PT_Load --[load_pt]--> TMA_Load --[load_q]---> MMA --[mma_s]--> Compute --[p_cor]--> Correction
                                        --[load_kv]-->     <--[p_mma]--
                                                           --[mma_o]--------------->

    :param schedule: MLA warp schedule defining warp role assignments.
    :param cluster_scale: Multiplier for cluster-scaled consumer/producer thread counts.
    """
    s = schedule
    load_tma = (s.load_tma_warp_id,)
    load_pt = (s.load_pt_warp_id,)
    mma = (s.mma_warp_id,)
    compute = s.compute_warp_ids
    correction = s.correction_warp_ids

    return PipelineTopology(
        edges=[
            PipelineEdge(
                "load_q",
                PipelineType.TMA_UMMA,
                stages=load_q_stages,
                producer_warp_ids=load_tma,
                consumer_warp_ids=mma,
                tx_count_key="q",
            ),
            PipelineEdge(
                "load_kv",
                PipelineType.TMA_UMMA,
                stages=load_kv_stages,
                producer_warp_ids=load_tma,
                consumer_warp_ids=mma,
                tx_count_key="kv",
            ),
            PipelineEdge(
                "mma_s",
                PipelineType.UMMA_ASYNC,
                stages=mma_s_stages,
                producer_warp_ids=mma,
                consumer_warp_ids=compute,
                cluster_scale=cluster_scale,
            ),
            PipelineEdge(
                "p_mma",
                PipelineType.ASYNC_UMMA,
                stages=p_mma_stages,
                producer_warp_ids=compute,
                consumer_warp_ids=mma,
                cluster_scale=cluster_scale,
            ),
            PipelineEdge(
                "p_cor",
                PipelineType.ASYNC,
                stages=p_cor_stages,
                producer_warp_ids=compute,
                consumer_warp_ids=correction,
            ),
            PipelineEdge(
                "mma_o",
                PipelineType.UMMA_ASYNC,
                stages=mma_o_stages,
                producer_warp_ids=mma,
                consumer_warp_ids=correction,
                cluster_scale=cluster_scale,
            ),
            PipelineEdge(
                "load_pt",
                PipelineType.CP_ASYNC,
                stages=load_pt_stages,
                producer_warp_ids=load_pt,
                consumer_warp_ids=load_tma,
            ),
        ]
    )


def make_mla_fp8_topology(
    schedule: MLAWarpScheduleFP8,
    load_q_stages: int = 1,
    load_k_stages: int = 3,
    load_v_stages: int = 2,
    mma_s_stages: int = 2,
    p_mma_stages: int = 2,
    p_cor_stages: int = 2,
    mma_o_stages: int = 2,
    cluster_scale: int = 2,
) -> PipelineTopology:
    """Build the pipeline topology for FP8 MLA decode.

    7 pipelines connecting 5 warp roles (no page-table pipeline)::

        TMA_K_Load --[load_q]---> MMA --[mma_s]--> Compute --[p_cor]--> Correction
                   --[load_k]-->     <--[p_mma]--
        TMA_V_Load --[load_v]-->     --[mma_o]--------------->

    FP8 splits the unified load_kv into separate load_k and load_v pipelines
    with dedicated TMA loader warps, and removes the page-table pipeline
    (page indices are read directly from global memory).

    :param schedule: FP8 MLA warp schedule defining warp role assignments.
    :param cluster_scale: Multiplier for cluster-scaled consumer/producer thread counts.
    """
    s = schedule
    load_k = (s.load_tma_k_warp_id,)
    load_v = (s.load_tma_v_warp_id,)
    mma = (s.mma_warp_id,)
    compute = s.compute_warp_ids
    correction = s.correction_warp_ids

    return PipelineTopology(
        edges=[
            PipelineEdge(
                "load_q",
                PipelineType.TMA_UMMA,
                stages=load_q_stages,
                producer_warp_ids=load_k,
                consumer_warp_ids=mma,
                tx_count_key="q",
            ),
            PipelineEdge(
                "load_k",
                PipelineType.TMA_UMMA,
                stages=load_k_stages,
                producer_warp_ids=load_k,
                consumer_warp_ids=mma,
                tx_count_key="kv",
            ),
            PipelineEdge(
                "load_v",
                PipelineType.TMA_UMMA,
                stages=load_v_stages,
                producer_warp_ids=load_v,
                consumer_warp_ids=mma,
                tx_count_key="vc",
            ),
            PipelineEdge(
                "mma_s",
                PipelineType.UMMA_ASYNC,
                stages=mma_s_stages,
                producer_warp_ids=mma,
                consumer_warp_ids=compute,
                cluster_scale=cluster_scale,
            ),
            PipelineEdge(
                "p_mma",
                PipelineType.ASYNC_UMMA,
                stages=p_mma_stages,
                producer_warp_ids=compute,
                consumer_warp_ids=mma,
                cluster_scale=cluster_scale,
            ),
            PipelineEdge(
                "p_cor",
                PipelineType.ASYNC,
                stages=p_cor_stages,
                producer_warp_ids=compute,
                consumer_warp_ids=correction,
            ),
            PipelineEdge(
                "mma_o",
                PipelineType.UMMA_ASYNC,
                stages=mma_o_stages,
                producer_warp_ids=mma,
                consumer_warp_ids=correction,
                cluster_scale=cluster_scale,
            ),
        ]
    )
