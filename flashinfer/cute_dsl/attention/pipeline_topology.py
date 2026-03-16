# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""PipelineTopology — declarative pipeline graph for attention kernels.

Replaces the imperative pipeline creation code (~76 lines of
make_pipeline_participants calls) with a declarative graph that can be
swapped between kernel variants (FMHA, MLA, decode).

Mirrors the C++ CUTLASS pattern where pipeline types are declared as
type aliases in the Mainloop collective, and the Kernel creates them.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any

import cutlass.pipeline as pipeline

from ..patch import pipeline as pipeline_patch
from .warp_schedule import WarpSchedule


class PipelineType(enum.Enum):
    """Pipeline types available in the CuTe DSL pipeline library.

    Thread count rules per type:
    - TMA_UMMA: leader-only (len(warps)) for both producer and consumer
    - UMMA_ASYNC: leader-only for producer, all-threads for consumer
    - ASYNC_UMMA: all-threads for producer, leader-only for consumer (reverse of UMMA_ASYNC)
    - ASYNC: all-threads for both producer and consumer
    """
    TMA_UMMA = "PipelineTmaUmma"
    UMMA_ASYNC = "PipelineUmmaAsync"
    ASYNC_UMMA = "PipelineAsyncUmma"
    ASYNC = "PipelineAsync"

    @property
    def cutlass_type(self):
        _map = {
            PipelineType.TMA_UMMA: pipeline.PipelineTmaUmma,
            PipelineType.UMMA_ASYNC: pipeline.PipelineUmmaAsync,
            PipelineType.ASYNC_UMMA: pipeline.PipelineAsyncUmma,
            PipelineType.ASYNC: pipeline.PipelineAsync,
        }
        return _map[self]

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

    When cluster_scale > 1 (e.g., 2-CTA MLA), the all-thread side of
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
    ) -> Dict[str, Tuple]:
        """Create all pipeline producer/consumer pairs from the topology.

        Uses the FlashInfer pipeline_patch API (for FMHA prefill).

        :param barrier_ptrs: Map from edge name to barrier storage pointer.
        :param tx_counts: Map from tx_count_key to byte count (for TMA pipelines).
        :param threads_per_warp: Threads per warp (typically 32).
        :returns: Dict mapping edge name to (producer, consumer) tuple.
        """
        result = {}
        for edge in self.edges:
            tx = None
            if edge.tx_count_key is not None:
                tx = tx_counts[edge.tx_count_key]

            prod_threads = edge.pipeline_type.producer_thread_count(
                len(edge.producer_warp_ids), threads_per_warp
            )
            cons_threads = edge.pipeline_type.consumer_thread_count(
                len(edge.consumer_warp_ids), threads_per_warp
            )

            producer, consumer = pipeline_patch.make_pipeline_participants(
                pipeline_type=edge.pipeline_type.cutlass_type,
                barrier_storage=barrier_ptrs[edge.name],
                num_stages=edge.stages,
                producer_thread_count=prod_threads,
                consumer_thread_count=cons_threads,
                tx_count=tx,
            )
            result[edge.name] = (producer, consumer)
        return result

    def create_pipelines_native(
        self,
        barrier_ptrs: Dict[str, Any],
        tx_counts: Dict[str, int],
        threads_per_warp: int,
        cta_layout_vmnk: Any = None,
    ) -> Dict[str, Any]:
        """Create all pipelines using the native cutlass.pipeline API.

        Used by MLA decode which needs cta_layout_vmnk for 2-CTA cluster
        support and uses PipelineAsyncUmma (not available via pipeline_patch).

        Returns a dict of pipeline objects (not producer/consumer tuples).
        Each pipeline is created via PipelineType.cutlass_type.create().

        :param barrier_ptrs: Map from edge name to barrier storage pointer.
        :param tx_counts: Map from tx_count_key to byte count (for TMA pipelines).
        :param threads_per_warp: Threads per warp (typically 32).
        :param cta_layout_vmnk: CTA layout for 2-CTA cluster support.
        :returns: Dict mapping edge name to pipeline object.
        """
        result = {}
        for edge in self.edges:
            # Compute thread counts
            prod_count = edge.pipeline_type.producer_thread_count(
                len(edge.producer_warp_ids), threads_per_warp
            )
            cons_count = edge.pipeline_type.consumer_thread_count(
                len(edge.consumer_warp_ids), threads_per_warp
            )

            # Apply cluster scaling to the all-thread side only.
            # UMMA_ASYNC: producer is leader-only, consumer is all-threads (scale consumer).
            # ASYNC_UMMA: producer is all-threads, consumer is leader-only (scale producer).
            if edge.cluster_scale > 1:
                if edge.pipeline_type == PipelineType.UMMA_ASYNC:
                    cons_count *= edge.cluster_scale
                elif edge.pipeline_type == PipelineType.ASYNC_UMMA:
                    prod_count *= edge.cluster_scale

            producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, prod_count,
                *([prod_count] if prod_count > 1 else []),
            )
            consumer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, cons_count,
                *([cons_count] if cons_count > 1 else []),
            )

            kwargs = {
                "barrier_storage": barrier_ptrs[edge.name],
                "num_stages": edge.stages,
                "producer_group": producer_group,
                "consumer_group": consumer_group,
            }
            if edge.tx_count_key is not None:
                kwargs["tx_count"] = tx_counts[edge.tx_count_key]
            if cta_layout_vmnk is not None and edge.pipeline_type != PipelineType.ASYNC:
                kwargs["cta_layout_vmnk"] = cta_layout_vmnk

            result[edge.name] = edge.pipeline_type.cutlass_type.create(**kwargs)
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

    return PipelineTopology(edges=[
        PipelineEdge("load_q", PipelineType.TMA_UMMA, stages=q_stages,
                     producer_warp_ids=load, consumer_warp_ids=mma,
                     tx_count_key="q"),
        PipelineEdge("load_kv", PipelineType.TMA_UMMA, stages=kv_stages,
                     producer_warp_ids=load, consumer_warp_ids=mma,
                     tx_count_key="kv"),
        PipelineEdge("mma_s0", PipelineType.UMMA_ASYNC, stages=mma_softmax_stages,
                     producer_warp_ids=mma, consumer_warp_ids=s0),
        PipelineEdge("mma_s1", PipelineType.UMMA_ASYNC, stages=mma_softmax_stages,
                     producer_warp_ids=mma, consumer_warp_ids=s1),
        PipelineEdge("s0_corr", PipelineType.ASYNC, stages=softmax_corr_stages,
                     producer_warp_ids=s0, consumer_warp_ids=corr),
        PipelineEdge("s1_corr", PipelineType.ASYNC, stages=softmax_corr_stages,
                     producer_warp_ids=s1, consumer_warp_ids=corr),
        PipelineEdge("corr_epi", PipelineType.ASYNC, stages=epi_stages,
                     producer_warp_ids=corr, consumer_warp_ids=epi),
        PipelineEdge("mma_corr", PipelineType.UMMA_ASYNC, stages=mma_corr_stages,
                     producer_warp_ids=mma, consumer_warp_ids=corr),
        PipelineEdge("s0_s1_sequence", PipelineType.ASYNC, stages=1,
                     producer_warp_ids=s0, consumer_warp_ids=s1),
    ])


def make_mla_topology(
    schedule: "MLAWarpSchedule",
    load_q_stages: int,
    load_kv_stages: int,
    mma_s_stages: int = 2,
    p_mma_stages: int = 2,
    mma_o_stages: int = 1,
    cluster_scale: int = 1,
) -> PipelineTopology:
    """Build the pipeline topology for MLA decode.

    The MLA decode kernel has 5 pipelines connecting 3 warp roles::

        Load --[load_q]--> MMA --[mma_s]--> Compute
              --[load_kv]->     <--[p_mma]--
                                --[mma_o]--> Compute

    :param schedule: MLA warp schedule defining warp role assignments.
    :param load_q_stages: Stage count for Q pipeline (= iterations_qk).
    :param load_kv_stages: Stage count for KV pipeline (= 24 // dtype_bytes).
    :param mma_s_stages: Stage count for MMA→Compute S pipeline.
    :param p_mma_stages: Stage count for Compute→MMA P pipeline.
    :param mma_o_stages: Stage count for MMA→Compute O pipeline.
    :param cluster_scale: Cluster size for thread count scaling (e.g., 2 for 2-CTA).
    """
    s = schedule
    load = (s.load_tma_warp_id,)
    mma = (s.mma_warp_id,)
    compute = s.compute_warp_ids

    return PipelineTopology(edges=[
        PipelineEdge("load_q", PipelineType.TMA_UMMA, stages=load_q_stages,
                     producer_warp_ids=load, consumer_warp_ids=mma,
                     tx_count_key="q"),
        PipelineEdge("load_kv", PipelineType.TMA_UMMA, stages=load_kv_stages,
                     producer_warp_ids=load, consumer_warp_ids=mma,
                     tx_count_key="kv"),
        PipelineEdge("mma_s", PipelineType.UMMA_ASYNC, stages=mma_s_stages,
                     producer_warp_ids=mma, consumer_warp_ids=compute,
                     cluster_scale=cluster_scale),
        PipelineEdge("p_mma", PipelineType.ASYNC_UMMA, stages=p_mma_stages,
                     producer_warp_ids=compute, consumer_warp_ids=mma,
                     cluster_scale=cluster_scale),
        PipelineEdge("mma_o", PipelineType.UMMA_ASYNC, stages=mma_o_stages,
                     producer_warp_ids=mma, consumer_warp_ids=compute,
                     cluster_scale=cluster_scale),
    ])
