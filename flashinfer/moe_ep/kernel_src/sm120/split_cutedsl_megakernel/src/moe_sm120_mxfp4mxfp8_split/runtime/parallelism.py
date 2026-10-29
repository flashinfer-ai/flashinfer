# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Package-local DP/TP/EP bootstrap for NUMA-local MegaMoE worlds.

The global one-node ``torchrun`` world is laid out as contiguous EP worlds::

    global rank = ((dp_rank * tp_size + tp_rank) * ep_size) + ep_rank

``DP2 x TP1 x EP4`` and ``DP1 x TP2 x EP4`` therefore both form contiguous
EP rank groups, while TP groups connect matching EP ranks across TP worlds.
Physical locality is not inferred from that rank layout: bootstrap gathers
each selected GPU's PCI BDF and sysfs NUMA node and derives an explicit EP
peer transport mask.

Only this package uses the subgroup topology; the public ``src.bootstrap``
remains unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist

import nvshmem.core

from .gpu_topology import (
    EpTransportTopology,
    discover_gpu_topology,
    format_ep_topology,
    gather_ep_transport_topology,
)

try:
    from cuda.core import Device
except ImportError:
    from cuda.core.experimental import Device


@dataclass(frozen=True)
class ParallelContext:
    """Global launch coordinates, subgroup coordinates, and process groups."""

    local_rank: int
    global_rank: int
    global_world_size: int
    data_parallel_rank: int
    data_parallel_size: int
    tensor_parallel_rank: int
    tensor_parallel_size: int
    ep_rank: int
    ep_size: int
    ep_global_ranks: Tuple[int, ...]
    tp_global_ranks: Tuple[int, ...]
    ep_group: dist.ProcessGroup
    ep_control_group: dist.ProcessGroup
    tp_group: Optional[dist.ProcessGroup]
    device: Device
    ep_transport_topology: EpTransportTopology

def _validate_sizes(
    global_world_size: int,
    data_parallel_size: int,
    tensor_parallel_size: int,
) -> int:
    if data_parallel_size <= 0:
        raise ValueError(
            f"data_parallel_size must be positive, got {data_parallel_size}."
        )
    if tensor_parallel_size <= 0:
        raise ValueError(
            f"tensor_parallel_size must be positive, got {tensor_parallel_size}."
        )
    parallel_product = data_parallel_size * tensor_parallel_size
    if global_world_size % parallel_product != 0:
        raise ValueError(
            f"global world size {global_world_size} must be divisible by "
            f"data_parallel_size * tensor_parallel_size = "
            f"{data_parallel_size} * {tensor_parallel_size} = "
            f"{parallel_product}."
        )
    return global_world_size // parallel_product


def init_parallel_context(
    data_parallel_size: int,
    tensor_parallel_size: int,
) -> ParallelContext:
    """Create process groups and discover topology, but do not init NVSHMEM.

    Splitting process-group/topology bootstrap from NVSHMEM initialization is
    required because the real EP peer layout selects the transport preset and
    NVSHMEM environment. Every process still creates groups in identical order.
    """

    if dist.is_initialized():
        raise RuntimeError(
            "parallel bootstrap must run before torch.distributed is initialized."
        )

    required = ("LOCAL_RANK", "RANK", "WORLD_SIZE", "MASTER_ADDR")
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise RuntimeError(
            "DP/TP/EP bootstrap currently requires torchrun variables; "
            f"missing {missing}."
        )

    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    global_world_size = int(os.environ["WORLD_SIZE"])
    ep_size = _validate_sizes(
        global_world_size,
        data_parallel_size,
        tensor_parallel_size,
    )

    ep_world_index = global_rank // ep_size
    data_parallel_rank = ep_world_index // tensor_parallel_size
    tensor_parallel_rank = ep_world_index % tensor_parallel_size
    ep_rank = global_rank % ep_size

    torch.cuda.set_device(local_rank)
    device = Device(local_rank)
    device.set_current()

    # Keep a CPU-only global control group so every rank can create both the
    # contiguous EP groups and the strided TP groups.  Avoiding an otherwise
    # unused global NCCL communicator also leaves NVSHMEM's per-EP NCCL team
    # initialization independent of the torch global world.
    dist.init_process_group(backend="gloo")

    ep_group: Optional[dist.ProcessGroup] = None
    ep_control_group: Optional[dist.ProcessGroup] = None
    ep_global_ranks: Tuple[int, ...] = ()
    num_ep_worlds = data_parallel_size * tensor_parallel_size

    for world_index in range(num_ep_worlds):
        ranks = tuple(
            range(world_index * ep_size, (world_index + 1) * ep_size)
        )
        group = dist.new_group(ranks=list(ranks), backend="nccl")
        if global_rank in ranks:
            ep_group = group
            ep_global_ranks = ranks

    # A separate Gloo EP group retains the launch-safety monitored barriers
    # without accidentally synchronizing another DP/TP world.
    for world_index in range(num_ep_worlds):
        ranks = tuple(
            range(world_index * ep_size, (world_index + 1) * ep_size)
        )
        group = dist.new_group(ranks=list(ranks), backend="gloo")
        if global_rank in ranks:
            ep_control_group = group

    tp_group: Optional[dist.ProcessGroup] = None
    tp_global_ranks: Tuple[int, ...] = (global_rank,)
    if tensor_parallel_size > 1:
        for dp_rank in range(data_parallel_size):
            for local_ep_rank in range(ep_size):
                ranks = tuple(
                    (
                        (dp_rank * tensor_parallel_size + tp_rank) * ep_size
                        + local_ep_rank
                    )
                    for tp_rank in range(tensor_parallel_size)
                )
                group = dist.new_group(ranks=list(ranks), backend="nccl")
                if global_rank in ranks:
                    tp_group = group
                    tp_global_ranks = ranks

    if ep_group is None or ep_control_group is None:
        raise RuntimeError(
            f"rank {global_rank} did not join an EP process group."
        )
    if tensor_parallel_size > 1 and tp_group is None:
        raise RuntimeError(
            f"rank {global_rank} did not join a TP process group."
        )

    local_topology = discover_gpu_topology(local_rank, global_rank)
    ep_transport_topology = gather_ep_transport_topology(
        local_topology,
        ep_rank=ep_rank,
        ep_control_group=ep_control_group,
    )

    os.environ["MEGAMOE_DATA_PARALLEL_RANK"] = str(data_parallel_rank)
    os.environ["MEGAMOE_DATA_PARALLEL_SIZE"] = str(data_parallel_size)
    os.environ["MEGAMOE_TENSOR_PARALLEL_RANK"] = str(tensor_parallel_rank)
    os.environ["MEGAMOE_TENSOR_PARALLEL_SIZE"] = str(tensor_parallel_size)
    os.environ["MEGAMOE_EP_RANK"] = str(ep_rank)
    os.environ["MEGAMOE_EP_SIZE"] = str(ep_size)

    if ep_rank == 0:
        print(
            "[DP/TP/EP topology] "
            + format_ep_topology(ep_transport_topology),
            flush=True,
        )

    return ParallelContext(
        local_rank=local_rank,
        global_rank=global_rank,
        global_world_size=global_world_size,
        data_parallel_rank=data_parallel_rank,
        data_parallel_size=data_parallel_size,
        tensor_parallel_rank=tensor_parallel_rank,
        tensor_parallel_size=tensor_parallel_size,
        ep_rank=ep_rank,
        ep_size=ep_size,
        ep_global_ranks=ep_global_ranks,
        tp_global_ranks=tp_global_ranks,
        ep_group=ep_group,
        ep_control_group=ep_control_group,
        tp_group=tp_group,
        device=device,
        ep_transport_topology=ep_transport_topology,
    )


def init_nvshmem(context: ParallelContext) -> ParallelContext:
    """Initialize the package-local EP NVSHMEM world after transport setup."""

    uid = nvshmem.core.get_unique_id(empty=(context.ep_rank != 0))
    uid_bytes = uid._data.view(np.uint8).copy()
    uid_tensor = torch.from_numpy(uid_bytes).cuda()
    dist.broadcast(
        uid_tensor,
        src=context.ep_global_ranks[0],
        group=context.ep_group,
    )
    dist.barrier(group=context.ep_group)
    uid._data[:] = uid_tensor.cpu().numpy().view(uid._data.dtype)

    nvshmem.core.init(
        device=context.device,
        uid=uid,
        rank=context.ep_rank,
        nranks=context.ep_size,
        initializer_method="uid",
    )
    print(
        "[DP/TP/EP bootstrap] "
        f"global_rank={context.global_rank}/{context.global_world_size} "
        f"local_rank={context.local_rank} "
        f"DP={context.data_parallel_rank}/{context.data_parallel_size} "
        f"TP={context.tensor_parallel_rank}/{context.tensor_parallel_size} "
        f"EP={context.ep_rank}/{context.ep_size} "
        f"ep_ranks={context.ep_global_ranks} "
        f"tp_ranks={context.tp_global_ranks} nvshmem_world=ep",
        flush=True,
    )
    return context


def init_parallel_and_nvshmem(
    data_parallel_size: int,
    tensor_parallel_size: int,
) -> ParallelContext:
    """Compatibility wrapper for callers that do not need transport setup."""

    return init_nvshmem(
        init_parallel_context(data_parallel_size, tensor_parallel_size)
    )


__all__ = [
    "ParallelContext",
    "init_nvshmem",
    "init_parallel_context",
    "init_parallel_and_nvshmem",
]
