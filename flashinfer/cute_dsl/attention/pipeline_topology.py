# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    ) -> Dict[str, Tuple[PipelineProducer, PipelineConsumer]]:
        """Create all pipeline producer/consumer pairs from the topology.

        :param barrier_ptrs: Map from edge name to barrier storage pointer.
        :param tx_counts: Map from tx_count_key to byte count (for TMA pipelines).
        :param threads_per_warp: Threads per warp (typically 32).
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

            create_kwargs = {
                "barrier_storage": barrier_ptrs[edge.name],
                "num_stages": edge.stages,
                "producer_group": CooperativeGroup(Agent.Thread, prod_threads),
                "consumer_group": CooperativeGroup(Agent.Thread, cons_threads),
                "defer_sync": True,
            }
            if edge.tx_count_key is not None:
                create_kwargs["tx_count"] = tx_counts[edge.tx_count_key]

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
