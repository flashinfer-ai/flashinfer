"""Calibrate GLM-shaped PCIe collectives and DCP prefill decisions.

Run this module under ``torchrun``. It intentionally has no vLLM dependency:
callers provide the model geometry and CKV record size that determine the
collective payloads. The probe uses B12X's real PCIe DMA all-reduce and
a separate NCCL communicator for each contiguous DCP rank group.

Example::

    torchrun --standalone --nproc-per-node=8 \
      -m b12x.comm.pcie.overlap_probe \
      --tp-size 8 --dcp-size 4 --context-tokens 8192,65536,131072

The result is a calibration, not an end-to-end model benchmark. It measures:

* B12X DMA versus NCCL for TP all-reduce payloads;
* contention between TP DMA and a layer-ahead DCP CKV all-gather;
* the full query-split indexer phase, including exact candidate exchange,
  owner merge, and final int32 top-k gather.

The query-split comparison includes the compute it avoids. Transport-only
numbers must not be used to disable a compute-saving path.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class ProbeConfig:
    tp_size: int = 8
    dcp_size: int = 4
    hidden_size: int = 6144
    tp_rows: int = 8192
    allreduce_rows: tuple[int, ...] = (1, 8, 32, 128, 512, 2048, 8192)
    ckv_record_bytes: int = 656
    context_tokens: tuple[int, ...] = (8192, 65536, 131072)
    indexer_shards: int = 2
    indexer_stride: int = 4
    indexer_heads: int = 32
    indexer_topk: int = 2048
    warmup_iterations: int = 3
    active_iterations: int = 9
    minimum_overlap_gain: float = 0.02
    minimum_backend_gain: float = 0.02
    minimum_query_split_gain: float = 0.02
    maximum_overlap_regression: float = 0.01
    maximum_collective_slowdown: float = 4.0
    dma_wire_mode: str = "0"

    def validate(self) -> None:
        if self.tp_size < 2:
            raise ValueError("tp_size must be at least 2")
        if self.dcp_size < 1 or self.tp_size % self.dcp_size:
            raise ValueError("dcp_size must divide tp_size and be positive")
        if self.hidden_size <= 0 or self.tp_rows <= 0:
            raise ValueError("hidden_size and tp_rows must be positive")
        if not self.allreduce_rows or any(value <= 0 for value in self.allreduce_rows):
            raise ValueError("allreduce_rows must contain positive values")
        if max(self.allreduce_rows) > self.tp_rows:
            raise ValueError("allreduce_rows cannot exceed tp_rows")
        if self.ckv_record_bytes <= 0:
            raise ValueError("ckv_record_bytes must be positive")
        if not self.context_tokens or any(value <= 0 for value in self.context_tokens):
            raise ValueError("context_tokens must contain positive values")
        if self.warmup_iterations < 0 or self.active_iterations < 1:
            raise ValueError("invalid iteration counts")
        if (
            self.indexer_shards < 1
            or self.dcp_size % self.indexer_shards
            or self.tp_size % self.indexer_shards
        ):
            raise ValueError("indexer_shards must divide both dcp_size and tp_size")
        if self.tp_rows % self.query_split_size:
            raise ValueError("tp_rows must divide evenly across query-split ranks")
        if self.tp_rows % self.tp_size:
            raise ValueError(
                "tp_rows must divide evenly across tp_size for owner merge"
            )
        if self.query_split_rows % self.indexer_shards:
            raise ValueError(
                "query_split_rows must divide evenly across indexer_shards"
            )
        if self.indexer_stride <= 0 or self.indexer_heads <= 0:
            raise ValueError("indexer_stride and indexer_heads must be positive")
        if self.indexer_topk <= 0:
            raise ValueError("indexer_topk must be positive")
        for context_tokens in self.context_tokens:
            local_tokens = self.indexer_local_tokens(context_tokens)
            local_topk = self.indexer_local_topk(context_tokens)
            if self.indexer_topk > local_tokens * self.indexer_shards:
                raise ValueError(
                    f"indexer_topk={self.indexer_topk} exceeds the "
                    f"{local_tokens * self.indexer_shards} global candidates "
                    f"available at context_tokens={context_tokens}"
                )
            if local_topk not in (512, 1024, 2048):
                raise ValueError(
                    f"context_tokens={context_tokens} requires unsupported local "
                    f"indexer top-k {local_topk}; supported values are 512, 1024, 2048"
                )
        if not 0.0 <= self.minimum_overlap_gain < 1.0:
            raise ValueError("minimum_overlap_gain must be in [0, 1)")
        if not 0.0 <= self.minimum_backend_gain < 1.0:
            raise ValueError("minimum_backend_gain must be in [0, 1)")
        if not 0.0 <= self.minimum_query_split_gain < 1.0:
            raise ValueError("minimum_query_split_gain must be in [0, 1)")
        if not 0.0 <= self.maximum_overlap_regression < 1.0:
            raise ValueError("maximum_overlap_regression must be in [0, 1)")
        if self.maximum_collective_slowdown < 1.0:
            raise ValueError("maximum_collective_slowdown must be at least 1")
        if self.dma_wire_mode.strip().lower() not in (
            "",
            "0",
            "false",
            "off",
            "no",
        ):
            raise ValueError(
                "automatic calibration only supports lossless BF16 DMA; "
                "compressed DMA modes must be selected explicitly"
            )

    @property
    def tp_payload_bytes(self) -> int:
        return self.tp_rows * self.hidden_size * torch.bfloat16.itemsize

    @property
    def query_split_size(self) -> int:
        return self.tp_size // self.indexer_shards

    @property
    def query_split_rows(self) -> int:
        return self.tp_rows // self.query_split_size

    def allreduce_payload_bytes(self, rows: int) -> int:
        return rows * self.hidden_size * torch.bfloat16.itemsize

    def indexer_local_tokens(self, context_tokens: int) -> int:
        index_tokens = math.ceil(context_tokens / self.indexer_stride)
        return math.ceil(index_tokens / self.indexer_shards)

    def indexer_local_topk(self, context_tokens: int) -> int:
        return min(self.indexer_topk, self.indexer_local_tokens(context_tokens))

    def query_split_final_wire_bytes_per_rank(self) -> int:
        return (
            (self.query_split_rows // self.indexer_shards)
            * self.indexer_topk
            * torch.int32.itemsize
            * (self.tp_size - 1)
        )

    def baseline_candidate_wire_bytes_per_rank(self, context_tokens: int) -> int:
        return (
            self.tp_rows
            * 2
            * self.indexer_local_topk(context_tokens)
            * torch.int32.itemsize
            * (self.indexer_shards - 1)
        )

    def owner_candidate_wire_bytes_per_rank(self, context_tokens: int) -> int:
        return (
            self.query_split_rows
            * 2
            * self.indexer_local_topk(context_tokens)
            * torch.int32.itemsize
            * (self.indexer_shards - 1)
            // self.indexer_shards
        )

    def query_split_wire_bytes_per_rank(self, context_tokens: int) -> int:
        return (
            self.owner_candidate_wire_bytes_per_rank(context_tokens)
            + self.query_split_final_wire_bytes_per_rank()
        )

    def ckv_local_tokens(self, context_tokens: int) -> int:
        return math.ceil(context_tokens / self.dcp_size)

    def ckv_local_bytes(self, context_tokens: int) -> int:
        return self.ckv_local_tokens(context_tokens) * self.ckv_record_bytes

    def ckv_wire_bytes_per_rank(self, context_tokens: int) -> int:
        return self.ckv_local_bytes(context_tokens) * (self.dcp_size - 1)


@dataclass(frozen=True)
class TimingSummary:
    median_ms: float
    minimum_ms: float
    maximum_ms: float


@dataclass(frozen=True)
class ContextDecision:
    enable_overlap: bool
    reason: str
    serialized_ms: float
    concurrent_wall_ms: float
    overlap_gain: float
    tp_slowdown: float
    ckv_slowdown: float
    ckv_effective_wire_gbps: float


@dataclass(frozen=True)
class BackendDecision:
    backend: str
    reason: str
    nccl_ms: float
    dma_ms: float
    dma_gain: float


@dataclass(frozen=True)
class QuerySplitDecision:
    enable_query_split: bool
    reason: str
    full_ms: float
    split_ms: float
    split_gain: float
    wire_bytes_per_rank: int


@dataclass(frozen=True)
class QueryBuffers:
    """Contiguous views over shared storage for one context's local top-k."""

    local_topk: int
    candidate_width: int
    full_indices: torch.Tensor
    full_scores: torch.Tensor
    full_global_indices: torch.Tensor
    full_candidates: torch.Tensor
    full_gathered_candidates: torch.Tensor
    full_candidate_indices: torch.Tensor
    full_candidate_scores: torch.Tensor
    full_candidate_lengths: torch.Tensor
    local_indices: torch.Tensor
    local_scores: torch.Tensor
    local_global_indices: torch.Tensor
    owner_candidates: torch.Tensor
    owner_received_candidates: torch.Tensor
    owner_candidate_indices: torch.Tensor
    owner_candidate_scores: torch.Tensor
    owner_candidate_lengths: torch.Tensor


def summarize(values: Sequence[float]) -> TimingSummary:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    return TimingSummary(
        median_ms=float(statistics.median(values)),
        minimum_ms=float(min(values)),
        maximum_ms=float(max(values)),
    )


def decide_overlap(
    *,
    tp_isolated_ms: float,
    ckv_isolated_ms: float,
    concurrent_tp_ms: float,
    concurrent_ckv_ms: float,
    concurrent_wall_ms: float,
    ckv_wire_bytes_per_rank: int,
    minimum_overlap_gain: float,
    maximum_collective_slowdown: float,
) -> ContextDecision:
    values = (
        tp_isolated_ms,
        ckv_isolated_ms,
        concurrent_tp_ms,
        concurrent_ckv_ms,
        concurrent_wall_ms,
    )
    if any(value <= 0 or not math.isfinite(value) for value in values):
        raise ValueError("all timing values must be positive and finite")

    serialized_ms = tp_isolated_ms + ckv_isolated_ms
    overlap_gain = (serialized_ms - concurrent_wall_ms) / serialized_ms
    tp_slowdown = concurrent_tp_ms / tp_isolated_ms
    ckv_slowdown = concurrent_ckv_ms / ckv_isolated_ms
    ckv_effective_wire_gbps = ckv_wire_bytes_per_rank / (concurrent_ckv_ms * 1e-3) / 1e9

    if overlap_gain < minimum_overlap_gain:
        enable = False
        reason = "insufficient-overlap-gain"
    elif max(tp_slowdown, ckv_slowdown) > maximum_collective_slowdown:
        enable = False
        reason = "collective-contention"
    else:
        enable = True
        reason = "measured-overlap-benefit"

    return ContextDecision(
        enable_overlap=enable,
        reason=reason,
        serialized_ms=serialized_ms,
        concurrent_wall_ms=concurrent_wall_ms,
        overlap_gain=overlap_gain,
        tp_slowdown=tp_slowdown,
        ckv_slowdown=ckv_slowdown,
        ckv_effective_wire_gbps=ckv_effective_wire_gbps,
    )


def decide_tp_backend(
    *, nccl_ms: float, dma_ms: float, minimum_backend_gain: float
) -> BackendDecision:
    if any(value <= 0 or not math.isfinite(value) for value in (nccl_ms, dma_ms)):
        raise ValueError("backend timings must be positive and finite")
    dma_gain = (nccl_ms - dma_ms) / nccl_ms
    if dma_gain >= minimum_backend_gain:
        backend = "b12x-dma"
        reason = "measured-dma-benefit"
    else:
        backend = "nccl"
        reason = "no-material-dma-benefit"
    return BackendDecision(
        backend=backend,
        reason=reason,
        nccl_ms=nccl_ms,
        dma_ms=dma_ms,
        dma_gain=dma_gain,
    )


def dma_crossover_bytes(
    points: Sequence[tuple[int, BackendDecision]], *, consecutive_wins: int = 2
) -> int:
    """Return the first measured payload in a sustained DMA-winning tail."""
    if consecutive_wins < 1:
        raise ValueError("consecutive_wins must be positive")
    ordered = sorted(points, key=lambda item: item[0])
    for index, (payload_bytes, decision) in enumerate(ordered):
        tail = ordered[index:]
        if len(tail) < consecutive_wins:
            break
        if decision.backend == "b12x-dma" and all(
            item[1].backend == "b12x-dma" for item in tail
        ):
            return payload_bytes
    return 0


def decide_query_split(
    *,
    full_ms: float,
    split_ms: float,
    wire_bytes_per_rank: int,
    minimum_query_split_gain: float,
) -> QuerySplitDecision:
    if any(value <= 0 or not math.isfinite(value) for value in (full_ms, split_ms)):
        raise ValueError("query-split timings must be positive and finite")
    if wire_bytes_per_rank < 0:
        raise ValueError("wire_bytes_per_rank must be non-negative")
    split_gain = (full_ms - split_ms) / full_ms
    if split_gain >= minimum_query_split_gain:
        enable = True
        reason = "measured-full-phase-benefit"
    else:
        enable = False
        reason = "no-material-full-phase-benefit"
    return QuerySplitDecision(
        enable_query_split=enable,
        reason=reason,
        full_ms=full_ms,
        split_ms=split_ms,
        split_gain=split_gain,
        wire_bytes_per_rank=wire_bytes_per_rank,
    )


def query_split_crossover_tokens(
    points: Sequence[tuple[int, QuerySplitDecision]],
    *,
    consecutive_wins: int = 2,
) -> int:
    """Return the first context in a sustained query-split winning tail."""
    if consecutive_wins < 1:
        raise ValueError("consecutive_wins must be positive")
    ordered = sorted(points, key=lambda item: item[0])
    for index, (context_tokens, decision) in enumerate(ordered):
        tail = ordered[index:]
        if len(tail) < consecutive_wins:
            break
        if decision.enable_query_split and all(
            item[1].enable_query_split for item in tail
        ):
            return context_tokens
    return 0


def recommend_prefetch_depth(
    decisions: Sequence[ContextDecision],
    *,
    maximum_overlap_regression: float,
) -> int:
    """Enable one-layer prefetch when it helps and is safe across the ladder."""
    if not decisions:
        raise ValueError("at least one overlap decision is required")
    if not 0.0 <= maximum_overlap_regression < 1.0:
        raise ValueError("maximum_overlap_regression must be in [0, 1)")
    harmful = any(
        decision.reason == "collective-contention"
        or decision.overlap_gain < -maximum_overlap_regression
        for decision in decisions
    )
    beneficial = any(decision.enable_overlap for decision in decisions)
    return int(beneficial and not harmful)


def maximum_contiguous_enabled_context(
    points: Sequence[tuple[int, bool]],
) -> int:
    """Return the end of the enabled prefix in ascending context order."""
    maximum_safe = 0
    for context_tokens, enabled in sorted(points):
        if not enabled:
            break
        maximum_safe = context_tokens
    return maximum_safe


def _global_max(values: Sequence[float]) -> tuple[float, ...]:
    tensor = torch.tensor(tuple(values), dtype=torch.float64, device="cpu")
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tuple(float(value) for value in tensor.tolist())


def _topology_snapshot(device: torch.device) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    local_device = {
        "rank": dist.get_rank(),
        "visible_ordinal": device.index,
        "pci_address": (
            f"{properties.pci_domain_id:08x}:{properties.pci_bus_id:02x}:"
            f"{properties.pci_device_id:02x}.0"
        ),
        "uuid": str(properties.uuid),
        "name": properties.name,
    }
    rank_devices: list[dict[str, Any] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(rank_devices, local_device)
    if dist.get_rank() != 0:
        return {}

    def run(command: list[str]) -> str:
        try:
            return subprocess.run(
                command,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            ).stdout.strip()
        except OSError as exc:
            return f"unavailable: {exc}"

    return {
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
        "nvidia_visible_devices": os.getenv("NVIDIA_VISIBLE_DEVICES", ""),
        "rank_devices": rank_devices,
        "gpu_inventory": run(
            [
                "nvidia-smi",
                "--query-gpu=index,pci.bus_id,name",
                "--format=csv,noheader",
            ]
        ),
        "topology_matrix": run(["nvidia-smi", "topo", "-m"]),
    }


class CollectiveOverlapProbe:
    def __init__(self, config: ProbeConfig) -> None:
        config.validate()
        self.config = config
        self.rank = int(os.environ["RANK"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        if self.world_size != config.tp_size:
            raise ValueError(
                f"torchrun world size {self.world_size} != tp_size {config.tp_size}"
            )

        torch.cuda.set_device(self.local_rank)
        self.device = torch.device("cuda", self.local_rank)
        dist.init_process_group(backend="gloo")

        tp_ranks = list(range(config.tp_size))
        self.tp_group = dist.new_group(tp_ranks, backend="nccl")
        self.dcp_group = None
        for start in range(0, config.tp_size, config.dcp_size):
            ranks = list(range(start, start + config.dcp_size))
            group = dist.new_group(ranks, backend="nccl")
            if self.rank in ranks:
                self.dcp_group = group
        if self.dcp_group is None:
            raise RuntimeError("rank was not assigned to a DCP group")

        replicas = [
            tp_ranks[start : start + config.indexer_shards]
            for start in range(0, config.tp_size, config.indexer_shards)
        ]
        self.indexer_group = None
        for ranks in replicas:
            group = dist.new_group(ranks, backend="nccl")
            if self.rank in ranks:
                self.indexer_group = group
        if self.indexer_group is None:
            raise RuntimeError("rank was not assigned to an indexer-shard group")
        self.indexer_shard_rank = dist.get_rank(group=self.indexer_group)

        self.query_split_group = None
        for shard_rank in range(config.indexer_shards):
            ranks = [replica[shard_rank] for replica in replicas]
            group = dist.new_group(ranks, backend="nccl")
            if self.rank in ranks:
                self.query_split_group = group
        if self.query_split_group is None:
            raise RuntimeError("rank was not assigned to a query-split group")
        self.query_split_rank = dist.get_rank(group=self.query_split_group)

        from b12x.comm.pcie import DmaAllReduce

        self.tp_input = torch.full(
            (config.tp_rows, config.hidden_size),
            float(self.rank + 1) / config.tp_size,
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.tp_nccl_input = torch.empty_like(self.tp_input)
        max_context = max(config.context_tokens)
        max_local_bytes = config.ckv_local_bytes(max_context)
        dcp_rank = dist.get_rank(group=self.dcp_group)
        self.ckv_input = torch.full(
            (max_local_bytes,),
            dcp_rank + 1,
            dtype=torch.uint8,
            device=self.device,
        )
        self.ckv_output = torch.empty(
            (config.dcp_size * max_local_bytes,),
            dtype=torch.uint8,
            device=self.device,
        )
        self.main_stream = torch.cuda.current_stream(self.device)
        self.side_stream = torch.cuda.Stream(device=self.device)
        self.release_stream = torch.cuda.Stream(device=self.device)
        self.dma = DmaAllReduce(
            exchange_group=self.tp_group,
            device=self.device,
            max_bytes=config.tp_payload_bytes,
            fp8=config.dma_wire_mode,
        )
        self.last_tp_output: torch.Tensor | None = None
        self._initialize_query_split()

    def _initialize_query_split(self) -> None:
        from b12x.attention.dsa_indexer.paged import (
            index_topk_fp8,
            prepare_paged_indexer_metadata,
        )
        from b12x.attention.dsa_indexer.scratch import (
            INDEXER_SOURCE_LAYOUT_PAGED,
            B12XIndexerScratchCaps,
            plan_indexer_scratch,
        )

        self._index_topk_fp8 = index_topk_fp8
        config = self.config
        max_local_tokens = config.indexer_local_tokens(max(config.context_tokens))
        self.indexer_page_bytes = 64 * (128 + 4)
        self.indexer_max_pages = math.ceil(max_local_tokens / 64)
        self.index_k_cache = torch.zeros(
            (self.indexer_max_pages, self.indexer_page_bytes),
            dtype=torch.uint8,
            device=self.device,
        )
        quant_plane = self.index_k_cache[:, : 64 * 128].view(
            self.indexer_max_pages, 64, 128
        )
        fp8_max_byte = int(
            torch.tensor(448.0, dtype=torch.float8_e4m3fn).view(torch.uint8).item()
        )
        quant_plane[..., 0].fill_(fp8_max_byte)
        analytic_scores = (
            1.0
            + self.indexer_shard_rank
            + config.indexer_shards
            * torch.arange(
                self.indexer_max_pages * 64,
                dtype=torch.float32,
                device=self.device,
            )
        )
        self.index_k_cache[:, 64 * 128 :].view(torch.float32).copy_(
            (analytic_scores / 448.0).view(self.indexer_max_pages, 64)
        )

        q_source = torch.zeros(
            (config.tp_rows, config.indexer_heads, 128),
            dtype=torch.float32,
            device=self.device,
        )
        q_source[..., 0] = 1.0
        self.index_q = q_source.to(torch.float8_e4m3fn)
        self.index_weights = torch.full(
            (config.tp_rows, config.indexer_heads),
            1.0 / config.indexer_heads,
            dtype=torch.float32,
            device=self.device,
        )

        self.full_indices = torch.empty(
            (config.tp_rows, config.indexer_topk),
            dtype=torch.int32,
            device=self.device,
        )
        self.full_scores = torch.empty(
            (config.tp_rows, config.indexer_topk),
            dtype=torch.float32,
            device=self.device,
        )
        self.local_indices = torch.empty(
            (config.query_split_rows, config.indexer_topk),
            dtype=torch.int32,
            device=self.device,
        )
        self.local_scores = torch.empty(
            (config.query_split_rows, config.indexer_topk),
            dtype=torch.float32,
            device=self.device,
        )
        self.gathered_indices = torch.empty_like(self.full_indices)

        from b12x.attention.dsa_indexer.tiled_topk import run_row_topk

        self._run_row_topk = run_row_topk
        shards = config.indexer_shards
        candidate_width = shards * config.indexer_topk
        self.full_global_indices = torch.empty_like(self.full_indices)
        self.full_candidates = torch.empty(
            (config.tp_rows, 2, config.indexer_topk),
            dtype=torch.int32,
            device=self.device,
        )
        self.full_gathered_candidates = torch.empty(
            (config.tp_rows * shards, 2, config.indexer_topk),
            dtype=torch.int32,
            device=self.device,
        )
        self.full_candidate_indices = torch.empty(
            (config.tp_rows, candidate_width),
            dtype=torch.int32,
            device=self.device,
        )
        self.full_candidate_scores = torch.empty(
            (config.tp_rows, candidate_width),
            dtype=torch.float32,
            device=self.device,
        )
        self.full_candidate_lengths = torch.full(
            (config.tp_rows,),
            candidate_width,
            dtype=torch.int32,
            device=self.device,
        )
        self.full_merged_values = torch.empty_like(self.full_scores)
        self.full_merged_indices = torch.empty_like(self.full_indices)

        owner_rows = config.query_split_rows // shards
        self.owner_rows = owner_rows
        self.local_global_indices = torch.empty_like(self.local_indices)
        self.owner_candidates = torch.empty(
            (config.query_split_rows, 2, config.indexer_topk),
            dtype=torch.int32,
            device=self.device,
        )
        self.owner_received_candidates = torch.empty_like(self.owner_candidates)
        self.owner_candidate_indices = torch.empty(
            (owner_rows, candidate_width),
            dtype=torch.int32,
            device=self.device,
        )
        self.owner_candidate_scores = torch.empty(
            (owner_rows, candidate_width),
            dtype=torch.float32,
            device=self.device,
        )
        self.owner_candidate_lengths = torch.full(
            (owner_rows,),
            candidate_width,
            dtype=torch.int32,
            device=self.device,
        )
        self.owner_merged_values = torch.empty(
            (owner_rows, config.indexer_topk),
            dtype=torch.float32,
            device=self.device,
        )
        self.owner_merged_indices = torch.empty(
            (owner_rows, config.indexer_topk),
            dtype=torch.int32,
            device=self.device,
        )

        def make_plan(rows: int):
            plan = plan_indexer_scratch(
                B12XIndexerScratchCaps(
                    device=self.device,
                    source_layout=INDEXER_SOURCE_LAYOUT_PAGED,
                    num_q_heads=config.indexer_heads,
                    max_q_rows=rows,
                    max_page_table_width=self.indexer_max_pages,
                    topk=config.indexer_topk,
                    mode="prefill",
                    shared_page_table=True,
                    reserve_paged_logits=False,
                )
            )
            scratch = [
                torch.empty(shape, dtype=dtype, device=self.device)
                for shape, dtype in plan.shapes_and_dtypes()
            ]
            return plan, scratch

        # Context bindings share each plan's scratch. Probe phases are strictly
        # sequential, so exactly one binding may be active at a time.
        full_plan, full_scratch = make_plan(config.tp_rows)
        split_plan, split_scratch = make_plan(config.query_split_rows)
        self.query_bindings: dict[int, tuple[Any, Any]] = {}
        for context_tokens in config.context_tokens:
            local_tokens = config.indexer_local_tokens(context_tokens)
            active_pages = math.ceil(local_tokens / 64)
            base_page_table = torch.full(
                (1, self.indexer_max_pages),
                -1,
                dtype=torch.int32,
                device=self.device,
            )
            base_page_table[0, :active_pages] = torch.arange(
                active_pages, dtype=torch.int32, device=self.device
            )

            def make_binding(
                plan,
                scratch,
                rows: int,
                base_page_table: torch.Tensor = base_page_table,
                local_tokens: int = local_tokens,
            ):
                page_table = base_page_table.expand(rows, -1)
                seqlens = torch.full(
                    (rows,), local_tokens, dtype=torch.int32, device=self.device
                )
                metadata = prepare_paged_indexer_metadata(
                    real_page_table=page_table,
                    cache_seqlens_int32=seqlens,
                    expected_num_q_heads=config.indexer_heads,
                    shared_page_table=True,
                )
                return plan.bind(
                    scratch=scratch,
                    real_page_table=metadata.real_page_table,
                    cache_seqlens_int32=metadata.cache_seqlens_int32,
                    schedule_metadata=metadata.schedule_metadata,
                    expected_num_q_heads=config.indexer_heads,
                    shared_page_table=True,
                )

            self.query_bindings[context_tokens] = (
                make_binding(full_plan, full_scratch, config.tp_rows),
                make_binding(split_plan, split_scratch, config.query_split_rows),
            )

    @staticmethod
    def _storage_view(storage: torch.Tensor, *shape: int) -> torch.Tensor:
        elements = math.prod(shape)
        if elements > storage.numel():
            raise RuntimeError(
                f"query buffer requires {elements} elements, storage has "
                f"{storage.numel()}"
            )
        return storage.view(-1)[:elements].view(*shape)

    def _query_buffers(self, context_tokens: int) -> QueryBuffers:
        config = self.config
        local_topk = config.indexer_local_topk(context_tokens)
        candidate_width = config.indexer_shards * local_topk
        rows = config.tp_rows
        local_rows = config.query_split_rows
        owner_rows = self.owner_rows

        full_lengths = self._storage_view(self.full_candidate_lengths, rows)
        owner_lengths = self._storage_view(self.owner_candidate_lengths, owner_rows)
        full_lengths.fill_(candidate_width)
        owner_lengths.fill_(candidate_width)
        return QueryBuffers(
            local_topk=local_topk,
            candidate_width=candidate_width,
            full_indices=self._storage_view(self.full_indices, rows, local_topk),
            full_scores=self._storage_view(self.full_scores, rows, local_topk),
            full_global_indices=self._storage_view(
                self.full_global_indices, rows, local_topk
            ),
            full_candidates=self._storage_view(
                self.full_candidates, rows, 2, local_topk
            ),
            full_gathered_candidates=self._storage_view(
                self.full_gathered_candidates,
                rows * config.indexer_shards,
                2,
                local_topk,
            ),
            full_candidate_indices=self._storage_view(
                self.full_candidate_indices, rows, candidate_width
            ),
            full_candidate_scores=self._storage_view(
                self.full_candidate_scores, rows, candidate_width
            ),
            full_candidate_lengths=full_lengths,
            local_indices=self._storage_view(
                self.local_indices, local_rows, local_topk
            ),
            local_scores=self._storage_view(self.local_scores, local_rows, local_topk),
            local_global_indices=self._storage_view(
                self.local_global_indices, local_rows, local_topk
            ),
            owner_candidates=self._storage_view(
                self.owner_candidates, local_rows, 2, local_topk
            ),
            owner_received_candidates=self._storage_view(
                self.owner_received_candidates, local_rows, 2, local_topk
            ),
            owner_candidate_indices=self._storage_view(
                self.owner_candidate_indices, owner_rows, candidate_width
            ),
            owner_candidate_scores=self._storage_view(
                self.owner_candidate_scores, owner_rows, candidate_width
            ),
            owner_candidate_lengths=owner_lengths,
        )

    def _prepare(self) -> None:
        torch.cuda.synchronize(self.device)
        dist.barrier()

    def _ckv_views(self, context_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
        local_bytes = self.config.ckv_local_bytes(context_tokens)
        return (
            self.ckv_input[:local_bytes],
            self.ckv_output[: local_bytes * self.config.dcp_size],
        )

    def _launch_ckv(self, context_tokens: int) -> None:
        input_tensor, output_tensor = self._ckv_views(context_tokens)
        dist.all_gather_into_tensor(
            output_tensor,
            input_tensor,
            group=self.dcp_group,
            async_op=False,
        )

    @staticmethod
    def _elapsed(start: torch.cuda.Event, end: torch.cuda.Event) -> float:
        return float(start.elapsed_time(end))

    def measure_tp(self) -> tuple[float, ...]:
        self._prepare()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(self.main_stream):
            start.record()
            self.last_tp_output = self.dma.all_reduce(self.tp_input)
            end.record()
        end.synchronize()
        return _global_max((self._elapsed(start, end),))

    def measure_tp_backend(self, rows: int, *, backend: str) -> tuple[float, ...]:
        self._prepare()
        source = self.tp_input[:rows]
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(self.main_stream):
            if backend == "nccl":
                target = self.tp_nccl_input[:rows]
                target.copy_(source)
                start.record()
                dist.all_reduce(target, group=self.tp_group)
                end.record()
            elif backend == "dma":
                start.record()
                self.last_tp_output = self.dma.all_reduce(source)
                end.record()
            else:
                raise ValueError(f"unknown TP backend {backend!r}")
        end.synchronize()
        return _global_max((self._elapsed(start, end),))

    def _run_query_indexer(self, context_tokens: int, *, split: bool) -> None:
        full_binding, split_binding = self.query_bindings[context_tokens]
        config = self.config
        buffers = self._query_buffers(context_tokens)
        if not split:
            self._index_topk_fp8(
                q_fp8=self.index_q,
                weights=self.index_weights,
                index_k_cache=self.index_k_cache,
                binding=full_binding,
                topk=buffers.local_topk,
                expected_num_q_heads=config.indexer_heads,
                out_indices=buffers.full_indices,
                out_scores=buffers.full_scores,
            )
            self._merge_replicated_candidates(buffers)
            return

        start = self.query_split_rank * config.query_split_rows
        end = start + config.query_split_rows
        self._index_topk_fp8(
            q_fp8=self.index_q[start:end].contiguous(),
            weights=self.index_weights[start:end].contiguous(),
            index_k_cache=self.index_k_cache,
            binding=split_binding,
            topk=buffers.local_topk,
            expected_num_q_heads=config.indexer_heads,
            out_indices=buffers.local_indices,
            out_scores=buffers.local_scores,
        )
        self._merge_owner_candidates(buffers)

    def _globalize_candidate_indices(
        self, local_indices: torch.Tensor, global_indices: torch.Tensor
    ) -> None:
        torch.mul(
            local_indices,
            self.config.indexer_shards,
            out=global_indices,
        )
        global_indices.add_(self.indexer_shard_rank)
        global_indices.masked_fill_(local_indices < 0, -1)

    def _unpack_rank_major_candidates(
        self,
        gathered: torch.Tensor,
        candidate_indices: torch.Tensor,
        candidate_scores: torch.Tensor,
        *,
        rows: int,
        local_topk: int,
    ) -> None:
        config = self.config
        rank_major = gathered.view(
            config.indexer_shards,
            rows,
            2,
            local_topk,
        )
        candidate_indices.view(rows, config.indexer_shards, local_topk).copy_(
            rank_major[:, :, 0, :].permute(1, 0, 2)
        )
        candidate_scores.view(torch.int32).view(
            rows, config.indexer_shards, local_topk
        ).copy_(rank_major[:, :, 1, :].permute(1, 0, 2))

    def _merge_replicated_candidates(self, buffers: QueryBuffers) -> None:
        config = self.config
        self._globalize_candidate_indices(
            buffers.full_indices, buffers.full_global_indices
        )
        buffers.full_candidates[:, 0, :].copy_(buffers.full_global_indices)
        buffers.full_candidates[:, 1, :].copy_(buffers.full_scores.view(torch.int32))
        dist.all_gather_into_tensor(
            buffers.full_gathered_candidates,
            buffers.full_candidates,
            group=self.indexer_group,
        )
        self._unpack_rank_major_candidates(
            buffers.full_gathered_candidates,
            buffers.full_candidate_indices,
            buffers.full_candidate_scores,
            rows=config.tp_rows,
            local_topk=buffers.local_topk,
        )
        self._run_row_topk(
            row_logits=buffers.full_candidate_scores,
            lengths=buffers.full_candidate_lengths,
            topk=config.indexer_topk,
            output_values=self.full_merged_values,
            output_indices=self.full_merged_indices,
            output_gather_table=buffers.full_candidate_indices,
        )

    def _merge_owner_candidates(self, buffers: QueryBuffers) -> None:
        config = self.config
        self._globalize_candidate_indices(
            buffers.local_indices, buffers.local_global_indices
        )
        buffers.owner_candidates[:, 0, :].copy_(buffers.local_global_indices)
        buffers.owner_candidates[:, 1, :].copy_(buffers.local_scores.view(torch.int32))
        dist.all_to_all_single(
            buffers.owner_received_candidates,
            buffers.owner_candidates,
            group=self.indexer_group,
        )
        self._unpack_rank_major_candidates(
            buffers.owner_received_candidates,
            buffers.owner_candidate_indices,
            buffers.owner_candidate_scores,
            rows=self.owner_rows,
            local_topk=buffers.local_topk,
        )
        self._run_row_topk(
            row_logits=buffers.owner_candidate_scores,
            lengths=buffers.owner_candidate_lengths,
            topk=config.indexer_topk,
            output_values=self.owner_merged_values,
            output_indices=self.owner_merged_indices,
            output_gather_table=buffers.owner_candidate_indices,
        )
        dist.all_gather_into_tensor(
            self.gathered_indices,
            self.owner_merged_indices,
            group=self.tp_group,
        )

    def measure_query_indexer(
        self, context_tokens: int, *, split: bool
    ) -> tuple[float, ...]:
        self._prepare()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(self.main_stream):
            start.record()
            self._run_query_indexer(context_tokens, split=split)
            end.record()
        end.synchronize()
        return _global_max((self._elapsed(start, end),))

    def measure_ckv(self, context_tokens: int) -> tuple[float, ...]:
        self._prepare()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(self.side_stream):
            start.record()
            self._launch_ckv(context_tokens)
            end.record()
        end.synchronize()
        return _global_max((self._elapsed(start, end),))

    def measure_overlap(
        self, context_tokens: int, *, side_first: bool
    ) -> tuple[float, ...]:
        self._prepare()
        release = torch.cuda.Event(enable_timing=True)
        tp_start = torch.cuda.Event(enable_timing=True)
        tp_end = torch.cuda.Event(enable_timing=True)
        ckv_start = torch.cuda.Event(enable_timing=True)
        ckv_end = torch.cuda.Event(enable_timing=True)

        # Keep both worker streams behind one short GPU-side gate while the host
        # queues their collectives. The gate is outside the measured interval.
        with torch.cuda.stream(self.release_stream):
            torch.cuda._sleep(2_000_000)
            release.record()
        self.main_stream.wait_event(release)
        self.side_stream.wait_event(release)

        def launch_tp() -> None:
            with torch.cuda.stream(self.main_stream):
                tp_start.record()
                self.last_tp_output = self.dma.all_reduce(self.tp_input)
                tp_end.record()

        def launch_ckv() -> None:
            with torch.cuda.stream(self.side_stream):
                ckv_start.record()
                self._launch_ckv(context_tokens)
                ckv_end.record()

        first: Callable[[], None]
        second: Callable[[], None]
        if side_first:
            first, second = launch_ckv, launch_tp
        else:
            first, second = launch_tp, launch_ckv
        first()
        second()
        tp_end.synchronize()
        ckv_end.synchronize()

        tp_ms = self._elapsed(tp_start, tp_end)
        ckv_ms = self._elapsed(ckv_start, ckv_end)
        wall_ms = max(
            self._elapsed(release, tp_end),
            self._elapsed(release, ckv_end),
        )
        return _global_max((tp_ms, ckv_ms, wall_ms))

    def validate_outputs(self, context_tokens: int) -> None:
        self._prepare()
        self.last_tp_output = self.dma.all_reduce(self.tp_input)
        self._launch_ckv(context_tokens)
        torch.cuda.synchronize(self.device)

        expected_tp = sum(range(1, self.config.tp_size + 1)) / self.config.tp_size
        assert self.last_tp_output is not None
        tp_ok = bool(
            torch.isclose(
                self.last_tp_output.flatten()[0].float(),
                torch.tensor(expected_tp, device=self.device),
                rtol=0,
                atol=0.05,
            ).item()
        )
        local_bytes = self.config.ckv_local_bytes(context_tokens)
        ckv_ok = True
        for offset in range(self.config.dcp_size):
            start = offset * local_bytes
            chunk = self.ckv_output[start : start + local_bytes]
            ckv_ok = ckv_ok and bool(torch.all(chunk == offset + 1).item())
        status = torch.tensor(
            [int(not tp_ok or not ckv_ok)], dtype=torch.int32, device="cpu"
        )
        dist.all_reduce(status, op=dist.ReduceOp.MAX)
        if int(status.item()):
            raise RuntimeError("collective output validation failed")

    def validate_tp_backends(self) -> None:
        rows = max(self.config.allreduce_rows)
        self._prepare()
        source = self.tp_input[:rows]
        nccl = self.tp_nccl_input[:rows]
        nccl.copy_(source)
        dist.all_reduce(nccl, group=self.tp_group)
        dma = self.dma.all_reduce(source)
        torch.cuda.synchronize(self.device)
        torch.testing.assert_close(dma, nccl, rtol=0, atol=0)

    def validate_query_split(self, context_tokens: int) -> None:
        self._prepare()
        self._run_query_indexer(context_tokens, split=False)
        self._run_query_indexer(context_tokens, split=True)
        torch.cuda.synchronize(self.device)
        torch.testing.assert_close(
            torch.sort(self.gathered_indices, dim=1).values,
            torch.sort(self.full_merged_indices, dim=1).values,
            rtol=0,
            atol=0,
        )

    def close(self) -> None:
        self.dma.close()
        torch.cuda.synchronize(self.device)
        dist.destroy_process_group()


def _rotate(values: tuple[str, ...], offset: int) -> tuple[str, ...]:
    index = offset % len(values)
    return values[index:] + values[:index]


def run_probe(config: ProbeConfig) -> dict[str, Any] | None:
    probe = CollectiveOverlapProbe(config)
    topology = _topology_snapshot(probe.device)
    rank = probe.rank
    modes = ("tp", "ckv", "side_first", "tp_first")
    result: dict[str, Any] = {
        "config": asdict(config),
        "topology": topology,
        "provenance": {
            "command": os.getenv("B12X_PROBE_COMMAND", ""),
            "source_revision": os.getenv("B12X_PROBE_SOURCE_REVISION", ""),
            "source_worktree": os.getenv("B12X_PROBE_SOURCE_WORKTREE", ""),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
        "correctness": {
            "tp_dma_matches_nccl": False,
            "tp_and_ckv_collectives": False,
            "query_split_matches_replicated": False,
        },
        "tp_allreduce": [],
        "contexts": [],
        "query_split": [],
    }
    try:
        probe.validate_tp_backends()
        result["correctness"]["tp_dma_matches_nccl"] = True
        probe.validate_outputs(max(config.context_tokens))
        result["correctness"]["tp_and_ckv_collectives"] = True
        for context_tokens in config.context_tokens:
            probe.validate_query_split(context_tokens)
        result["correctness"]["query_split_matches_replicated"] = True

        backend_modes = ("nccl", "dma")
        backend_decisions: list[tuple[int, BackendDecision]] = []
        for row_index, rows in enumerate(config.allreduce_rows):
            backend_samples: dict[str, list[float]] = {
                mode: [] for mode in backend_modes
            }
            total_iterations = config.warmup_iterations + config.active_iterations
            for iteration in range(total_iterations):
                for mode in _rotate(backend_modes, row_index + iteration):
                    value = probe.measure_tp_backend(rows, backend=mode)[0]
                    if iteration >= config.warmup_iterations:
                        backend_samples[mode].append(value)
            nccl = summarize(backend_samples["nccl"])
            dma = summarize(backend_samples["dma"])
            payload_bytes = config.allreduce_payload_bytes(rows)
            decision = decide_tp_backend(
                nccl_ms=nccl.median_ms,
                dma_ms=dma.median_ms,
                minimum_backend_gain=config.minimum_backend_gain,
            )
            backend_decisions.append((payload_bytes, decision))
            if rank == 0:
                result["tp_allreduce"].append(
                    {
                        "rows": rows,
                        "payload_bytes": payload_bytes,
                        "nccl": asdict(nccl),
                        "dma": asdict(dma),
                        "samples_ms": backend_samples,
                        "decision": asdict(decision),
                    }
                )

        for context_index, context_tokens in enumerate(config.context_tokens):
            overlap_samples: dict[str, list[tuple[float, ...]]] = {
                mode: [] for mode in modes
            }
            total_iterations = config.warmup_iterations + config.active_iterations
            for iteration in range(total_iterations):
                for mode in _rotate(modes, context_index + iteration):
                    if mode == "tp":
                        values = probe.measure_tp()
                    elif mode == "ckv":
                        values = probe.measure_ckv(context_tokens)
                    else:
                        values = probe.measure_overlap(
                            context_tokens, side_first=mode == "side_first"
                        )
                    if iteration >= config.warmup_iterations:
                        overlap_samples[mode].append(values)

            tp = summarize([value[0] for value in overlap_samples["tp"]])
            ckv = summarize([value[0] for value in overlap_samples["ckv"]])
            side_tp = summarize([value[0] for value in overlap_samples["side_first"]])
            side_ckv = summarize([value[1] for value in overlap_samples["side_first"]])
            side_wall = summarize([value[2] for value in overlap_samples["side_first"]])
            tp_first_tp = summarize([value[0] for value in overlap_samples["tp_first"]])
            tp_first_ckv = summarize(
                [value[1] for value in overlap_samples["tp_first"]]
            )
            tp_first_wall = summarize(
                [value[2] for value in overlap_samples["tp_first"]]
            )

            decision = decide_overlap(
                tp_isolated_ms=tp.median_ms,
                ckv_isolated_ms=ckv.median_ms,
                concurrent_tp_ms=max(side_tp.median_ms, tp_first_tp.median_ms),
                concurrent_ckv_ms=max(side_ckv.median_ms, tp_first_ckv.median_ms),
                concurrent_wall_ms=max(side_wall.median_ms, tp_first_wall.median_ms),
                ckv_wire_bytes_per_rank=config.ckv_wire_bytes_per_rank(context_tokens),
                minimum_overlap_gain=config.minimum_overlap_gain,
                maximum_collective_slowdown=config.maximum_collective_slowdown,
            )
            if rank == 0:
                result["contexts"].append(
                    {
                        "context_tokens": context_tokens,
                        "ckv_local_tokens": config.ckv_local_tokens(context_tokens),
                        "ckv_local_bytes": config.ckv_local_bytes(context_tokens),
                        "ckv_wire_bytes_per_rank": config.ckv_wire_bytes_per_rank(
                            context_tokens
                        ),
                        "tp_isolated": asdict(tp),
                        "ckv_isolated": asdict(ckv),
                        "side_first": {
                            "tp": asdict(side_tp),
                            "ckv": asdict(side_ckv),
                            "wall": asdict(side_wall),
                        },
                        "tp_first": {
                            "tp": asdict(tp_first_tp),
                            "ckv": asdict(tp_first_ckv),
                            "wall": asdict(tp_first_wall),
                        },
                        "samples_ms": {
                            mode: [list(values) for values in overlap_samples[mode]]
                            for mode in modes
                        },
                        "decision": asdict(decision),
                    }
                )

            query_samples: dict[str, list[float]] = {"full": [], "split": []}
            query_modes = ("full", "split")
            for iteration in range(total_iterations):
                for mode in _rotate(query_modes, context_index + iteration):
                    value = probe.measure_query_indexer(
                        context_tokens, split=mode == "split"
                    )[0]
                    if iteration >= config.warmup_iterations:
                        query_samples[mode].append(value)
            full = summarize(query_samples["full"])
            split = summarize(query_samples["split"])
            query_decision = decide_query_split(
                full_ms=full.median_ms,
                split_ms=split.median_ms,
                wire_bytes_per_rank=config.query_split_wire_bytes_per_rank(
                    context_tokens
                ),
                minimum_query_split_gain=config.minimum_query_split_gain,
            )
            if rank == 0:
                result["query_split"].append(
                    {
                        "context_tokens": context_tokens,
                        "indexer_local_tokens": config.indexer_local_tokens(
                            context_tokens
                        ),
                        "indexer_local_topk": config.indexer_local_topk(context_tokens),
                        "query_split_size": config.query_split_size,
                        "full_rows": config.tp_rows,
                        "local_rows": config.query_split_rows,
                        "wire_bytes_per_rank": (
                            config.query_split_wire_bytes_per_rank(context_tokens)
                        ),
                        "wire_breakdown_bytes_per_rank": {
                            "baseline_candidate_allgather": (
                                config.baseline_candidate_wire_bytes_per_rank(
                                    context_tokens
                                )
                            ),
                            "owner_candidate_exchange": (
                                config.owner_candidate_wire_bytes_per_rank(
                                    context_tokens
                                )
                            ),
                            "final_topk_allgather": (
                                config.query_split_final_wire_bytes_per_rank()
                            ),
                        },
                        "full": asdict(full),
                        "split": asdict(split),
                        "samples_ms": query_samples,
                        "decision": asdict(query_decision),
                    }
                )

        if rank == 0:
            overlap_decisions = [
                ContextDecision(**row["decision"]) for row in result["contexts"]
            ]
            result["recommended_prefetch_depth"] = recommend_prefetch_depth(
                overlap_decisions,
                maximum_overlap_regression=config.maximum_overlap_regression,
            )
            result["maximum_safe_context_tokens"] = maximum_contiguous_enabled_context(
                [
                    (
                        row["context_tokens"],
                        row["decision"]["enable_overlap"],
                    )
                    for row in result["contexts"]
                ]
            )
            query_crossover = query_split_crossover_tokens(
                [
                    (
                        row["context_tokens"],
                        QuerySplitDecision(**row["decision"]),
                    )
                    for row in result["query_split"]
                ]
            )
            result["policy"] = {
                "numeric_contract": "lossless-only",
                "compressed_dma_requires_explicit_opt_in": True,
                "tp_allreduce": {
                    "dma_wire_mode": "bf16",
                    "dma_min_bytes": dma_crossover_bytes(backend_decisions),
                    "fallback": "nccl",
                },
                "dcp_ckv_prefetch_depth": result["recommended_prefetch_depth"],
                "dcp_query_split": int(query_crossover > 0),
                "dcp_query_split_min_context_tokens": query_crossover,
            }
            return result
        return None
    finally:
        probe.close()


def _parse_positive_ints(value: str) -> tuple[int, ...]:
    try:
        contexts = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not contexts or any(item <= 0 for item in contexts):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return contexts


def _build_parser() -> argparse.ArgumentParser:
    defaults = ProbeConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp-size", type=int, default=defaults.tp_size)
    parser.add_argument("--dcp-size", type=int, default=defaults.dcp_size)
    parser.add_argument("--hidden-size", type=int, default=defaults.hidden_size)
    parser.add_argument("--tp-rows", type=int, default=defaults.tp_rows)
    parser.add_argument(
        "--allreduce-rows",
        type=_parse_positive_ints,
        default=defaults.allreduce_rows,
    )
    parser.add_argument(
        "--ckv-record-bytes", type=int, default=defaults.ckv_record_bytes
    )
    parser.add_argument(
        "--context-tokens",
        type=_parse_positive_ints,
        default=defaults.context_tokens,
    )
    parser.add_argument("--indexer-shards", type=int, default=defaults.indexer_shards)
    parser.add_argument("--indexer-stride", type=int, default=defaults.indexer_stride)
    parser.add_argument("--indexer-heads", type=int, default=defaults.indexer_heads)
    parser.add_argument("--indexer-topk", type=int, default=defaults.indexer_topk)
    parser.add_argument(
        "--warmup-iterations", type=int, default=defaults.warmup_iterations
    )
    parser.add_argument(
        "--active-iterations", type=int, default=defaults.active_iterations
    )
    parser.add_argument(
        "--minimum-overlap-gain", type=float, default=defaults.minimum_overlap_gain
    )
    parser.add_argument(
        "--minimum-backend-gain", type=float, default=defaults.minimum_backend_gain
    )
    parser.add_argument(
        "--minimum-query-split-gain",
        type=float,
        default=defaults.minimum_query_split_gain,
    )
    parser.add_argument(
        "--maximum-overlap-regression",
        type=float,
        default=defaults.maximum_overlap_regression,
    )
    parser.add_argument(
        "--maximum-collective-slowdown",
        type=float,
        default=defaults.maximum_collective_slowdown,
    )
    parser.add_argument(
        "--dma-wire-mode",
        default=defaults.dma_wire_mode,
        help="automatic policy accepts only lossless BF16 mode 0",
    )
    parser.add_argument("--output", type=Path)
    return parser


def _print_summary(result: dict[str, Any]) -> None:
    print("TP all-reduce (lossless BF16)", flush=True)
    print("rows  bytes       nccl_ms  dma_ms  dma_gain  verdict", flush=True)
    for row in result["tp_allreduce"]:
        decision = row["decision"]
        print(
            f"{row['rows']:>4}  {row['payload_bytes']:>10}  "
            f"{row['nccl']['median_ms']:>7.3f}  "
            f"{row['dma']['median_ms']:>6.3f}  "
            f"{decision['dma_gain']:>8.1%}  {decision['backend']}",
            flush=True,
        )
    print("\nCKV prefetch overlap", flush=True)
    print(
        "context  tp_iso  ckv_iso  pair_wall  gain    tp_x  ckv_x  verdict",
        flush=True,
    )
    for row in result["contexts"]:
        decision = row["decision"]
        print(
            f"{row['context_tokens']:>7}  "
            f"{row['tp_isolated']['median_ms']:>6.2f}  "
            f"{row['ckv_isolated']['median_ms']:>7.2f}  "
            f"{decision['concurrent_wall_ms']:>9.2f}  "
            f"{decision['overlap_gain']:>6.1%}  "
            f"{decision['tp_slowdown']:>4.2f}  "
            f"{decision['ckv_slowdown']:>5.2f}  "
            f"{decision['reason']}",
            flush=True,
        )
    print(
        "recommended_prefetch_depth="
        f"{result['recommended_prefetch_depth']} "
        "maximum_safe_context_tokens="
        f"{result['maximum_safe_context_tokens']}",
        flush=True,
    )
    print(
        "\nQuery split: indexer + exact owner candidate merge + final gather",
        flush=True,
    )
    print("context  full_ms  split_ms  gain    verdict", flush=True)
    for row in result["query_split"]:
        decision = row["decision"]
        print(
            f"{row['context_tokens']:>7}  "
            f"{row['full']['median_ms']:>7.2f}  "
            f"{row['split']['median_ms']:>8.2f}  "
            f"{decision['split_gain']:>6.1%}  {decision['reason']}",
            flush=True,
        )
    print("policy=" + json.dumps(result["policy"], sort_keys=True), flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = ProbeConfig(
        tp_size=args.tp_size,
        dcp_size=args.dcp_size,
        hidden_size=args.hidden_size,
        tp_rows=args.tp_rows,
        allreduce_rows=args.allreduce_rows,
        ckv_record_bytes=args.ckv_record_bytes,
        context_tokens=args.context_tokens,
        indexer_shards=args.indexer_shards,
        indexer_stride=args.indexer_stride,
        indexer_heads=args.indexer_heads,
        indexer_topk=args.indexer_topk,
        warmup_iterations=args.warmup_iterations,
        active_iterations=args.active_iterations,
        minimum_overlap_gain=args.minimum_overlap_gain,
        minimum_backend_gain=args.minimum_backend_gain,
        minimum_query_split_gain=args.minimum_query_split_gain,
        maximum_overlap_regression=args.maximum_overlap_regression,
        maximum_collective_slowdown=args.maximum_collective_slowdown,
        dma_wire_mode=args.dma_wire_mode,
    )
    try:
        config.validate()
    except ValueError as exc:
        print(f"invalid probe configuration: {exc}", file=sys.stderr)
        return 2
    if "RANK" not in os.environ:
        print(
            "This probe must run under torchrun; see --help for an example.",
            file=sys.stderr,
        )
        return 2
    result = run_probe(config)
    if result is not None:
        _print_summary(result)
        encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded, encoding="utf-8")
        else:
            print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
