# Copyright 2026 Cursor Research
# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Caller-owned symmetric memory and metadata for MoK training.

Adapted from cursor/mixture-of-kittens caeb2963f855c7ad53bb50c8bbf211086405cb98.
Modified: standalone host-only dependency closure and device admission.
"""

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem


@dataclass(frozen=True, slots=True)
class MoKConfig:
    fwd_num_comm_sms: int = 40
    bwd_num_comm_sms: int = 28
    minibatch_size: int = 4096
    macrobatch_size: int = 131072
    schedule_capacity_multiplier: float = 0.5
    all_gather_top_experts_chunk_bytes: int = 2048


@dataclass(frozen=True, slots=True)
class MoKSchedule:
    peer_rank: torch.Tensor  # (schedule_capacity,) int32
    peer_token_idx: torch.Tensor  # (schedule_capacity,) int32
    num_tokens: torch.Tensor  # (1,) int32
    tokens_per_expert: torch.Tensor  # (num_local_experts,) int32


@dataclass(frozen=True, slots=True)
class MoKForwardContext:
    x_routed: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    gate_shared: torch.Tensor
    gate_routed: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    up_shared: torch.Tensor
    up_routed: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    hidden_shared: torch.Tensor
    hidden_routed: torch.Tensor | tuple[torch.Tensor, torch.Tensor]


@dataclass(slots=True)
class MoKWorkspace:
    group_name: str  # () str
    ep_rank: int  # () int
    ep_size: int  # () int
    device: torch.device  # () torch.device
    num_local_tokens: int  # () int
    hidden_size: int  # () int
    topk: int  # () int
    schedule_capacity: int  # () int
    x_buffer: torch.Tensor  # (num_local_tokens, hidden_size) bfloat16
    x_buffer_handle: Any  # () SymmetricMemory handle
    x_buffer_ptrs: list[int]  # (ep_size,) uintptr64
    combine_buffer: torch.Tensor  # (num_local_tokens * topk, hidden_size) bfloat16
    combine_buffer_handle: Any  # () SymmetricMemory handle
    combine_buffer_ptrs: list[int]  # (ep_size,) uintptr64
    d_y_buffer: torch.Tensor  # (num_local_tokens, hidden_size) bfloat16
    d_y_buffer_handle: Any  # () SymmetricMemory handle
    d_y_buffer_ptrs: list[int]  # (ep_size,) uintptr64
    d_x_routed_buffer: torch.Tensor  # (num_local_tokens * topk, hidden_size) bfloat16
    d_x_routed_buffer_handle: Any  # () SymmetricMemory handle
    d_x_routed_buffer_ptrs: list[int]  # (ep_size,) uintptr64
    router_weight_buffer: torch.Tensor  # (num_local_tokens, topk) float32
    router_weight_buffer_handle: Any  # () SymmetricMemory handle
    router_weight_buffer_ptrs: list[int]  # (ep_size,) uintptr64
    d_router_weight_buffer: torch.Tensor  # (num_local_tokens, topk) float32
    d_router_weight_buffer_handle: Any  # () SymmetricMemory handle
    d_router_weight_buffer_ptrs: list[int]  # (ep_size,) uintptr64
    all_gather_top_experts_buffer: (
        torch.Tensor
    )  # (ep_size, num_local_tokens, topk) int32
    all_gather_top_experts_buffer_handle: Any  # () SymmetricMemory handle
    all_gather_top_experts_buffer_multicast_ptr: int  # () uintptr64
    barrier_buffer: torch.Tensor  # (1,) int32
    barrier_buffer_handle: Any  # () SymmetricMemory handle
    barrier_buffer_ptrs: list[int]  # (ep_size,) uintptr64
    barrier_buffer_multicast_ptr: int  # () uintptr64
    barrier_target: torch.Tensor  # (1,) int32


def validate_workspace_args(
    config: MoKConfig,
    group: dist.ProcessGroup,
    *,
    device: torch.device,
    num_local_tokens: int,
    hidden_size: int,
    topk: int,
) -> None:
    """Validates the arguments used to create or retrieve a workspace.

    Inputs:
        config:           MoKConfig
        group:            torch.distributed.ProcessGroup
        device:           torch.device
        num_local_tokens: int
        hidden_size:      int
        topk:             int

    Outputs:
        None
    """
    if (
        type(config.schedule_capacity_multiplier) not in (int, float)
        or not math.isfinite(config.schedule_capacity_multiplier)
        or config.schedule_capacity_multiplier <= 0
    ):
        raise ValueError(
            "schedule_capacity_multiplier must be a positive finite number"
        )
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")
    if not isinstance(group, dist.ProcessGroup):
        raise TypeError("group must be a torch.distributed.ProcessGroup")
    if not isinstance(device, torch.device):
        raise TypeError("device must be a torch.device")
    if device.type != "cuda":
        raise ValueError("device must be a CUDA device")
    device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    device = torch.device("cuda", device_index)
    if device_index != torch.cuda.current_device():
        raise ValueError("MoK workspace device must be the current CUDA device")
    if torch.cuda.get_device_capability(device) not in ((10, 0), (10, 3), (10, 7)):
        raise NotImplementedError("MoK requires an SM100a-compatible device")
    device_properties = torch.cuda.get_device_properties(device)
    if type(num_local_tokens) is not int or num_local_tokens < 512:
        raise ValueError("num_local_tokens must be an integer at least 512")
    if num_local_tokens % 256 != 0:
        raise ValueError("num_local_tokens must be divisible by 256")
    if type(hidden_size) is not int or hidden_size <= 0:
        raise ValueError("hidden_size must be a positive integer")
    if hidden_size % 256 != 0:
        raise ValueError("hidden_size must be divisible by 256")
    if type(topk) is not int or not 0 < topk <= 255:
        raise ValueError("topk must be an integer in [1, 255]")
    # Six-stage ring of (top-k + 1) 1024-column BF16 rows, plus barriers.
    fwd_epilogue_smem_bytes = 6 * (topk + 1) * 2048 + 1024
    if fwd_epilogue_smem_bytes > device_properties.shared_memory_per_block_optin:
        raise ValueError(
            "topk requires more dynamic shared memory than the device supports"
        )

    group_name = group.group_name
    if not isinstance(group_name, str) or not group_name:
        raise RuntimeError("process group must have a nonempty group_name")
    ep_rank = dist.get_rank(group=group)
    ep_size = dist.get_world_size(group=group)
    if ep_size not in (1, 4, 8, 16, 32, 64):
        raise ValueError("MoK EP size must be one of 1, 4, 8, 16, 32, 64")
    if not 0 <= ep_rank < ep_size:
        raise RuntimeError("current process is not a member of the EP process group")


def create_workspace(
    config: MoKConfig,
    group: dist.ProcessGroup,
    *,
    device: torch.device,
    num_local_tokens: int,
    hidden_size: int,
    topk: int,
) -> MoKWorkspace:
    """Creates a new caller-owned workspace.

    Inputs:
        config:           MoKConfig
        group:            torch.distributed.ProcessGroup
        device:           torch.device
        num_local_tokens: int
        hidden_size:      int
        topk:             int

    Outputs:
        workspace: MoKWorkspace
    """
    validate_workspace_args(
        config,
        group,
        device=device,
        num_local_tokens=num_local_tokens,
        hidden_size=hidden_size,
        topk=topk,
    )

    device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    device = torch.device("cuda", device_index)
    group_name = group.group_name
    ep_rank = dist.get_rank(group=group)
    ep_size = dist.get_world_size(group=group)
    schedule_capacity_factor = max(
        2, math.ceil(ep_size * config.schedule_capacity_multiplier)
    )

    local_shape = torch.tensor(
        [num_local_tokens, hidden_size, topk], dtype=torch.int64, device=device
    )
    gathered_shapes = torch.empty(
        ep_size * local_shape.numel(), dtype=torch.int64, device=device
    )
    dist.all_gather_into_tensor(gathered_shapes, local_shape, group=group)
    gathered_shapes = gathered_shapes.view(ep_size, local_shape.numel())
    if not torch.all(
        gathered_shapes == local_shape
    ).item():  # .item() here is fine since this is one-time setup
        raise ValueError(
            "MoK requires identical token, hidden, and top-k shapes on every EP rank"
        )
    if ep_size > 1:
        symm_mem.enable_symm_mem_for_group(group_name)

    schedule_capacity = num_local_tokens * topk * schedule_capacity_factor

    def allocate_buffer(
        *shape: int, dtype: torch.dtype, zero: bool = False
    ) -> tuple[torch.Tensor, Any, list[int]]:
        if ep_size == 1:
            if zero:
                buffer = torch.zeros(*shape, dtype=dtype, device=device)
            else:
                buffer = torch.empty(*shape, dtype=dtype, device=device)
            handle = None
            ptrs = [int(buffer.data_ptr())]
        else:
            buffer = symm_mem.empty(*shape, dtype=dtype, device=device)
            if zero:
                buffer.zero_()
            handle = symm_mem.rendezvous(buffer, group_name)
            ptrs = [int(handle.buffer_ptrs[peer_rank]) for peer_rank in range(ep_size)]
        return buffer, handle, ptrs

    x_buffer, x_buffer_handle, x_buffer_ptrs = allocate_buffer(
        num_local_tokens, hidden_size, dtype=torch.bfloat16
    )
    combine_buffer, combine_buffer_handle, combine_buffer_ptrs = allocate_buffer(
        num_local_tokens * topk, hidden_size, dtype=torch.bfloat16
    )
    d_y_buffer, d_y_buffer_handle, d_y_buffer_ptrs = allocate_buffer(
        num_local_tokens, hidden_size, dtype=torch.bfloat16
    )
    d_x_routed_buffer, d_x_routed_buffer_handle, d_x_routed_buffer_ptrs = (
        allocate_buffer(num_local_tokens * topk, hidden_size, dtype=torch.bfloat16)
    )
    router_weight_buffer, router_weight_buffer_handle, router_weight_buffer_ptrs = (
        allocate_buffer(num_local_tokens, topk, dtype=torch.float32)
    )
    (
        d_router_weight_buffer,
        d_router_weight_buffer_handle,
        d_router_weight_buffer_ptrs,
    ) = allocate_buffer(num_local_tokens, topk, dtype=torch.float32)

    all_gather_top_experts_buffer, all_gather_top_experts_buffer_handle, _ = (
        allocate_buffer(ep_size, num_local_tokens, topk, dtype=torch.int32)
    )
    all_gather_top_experts_buffer_multicast_ptr = int(
        all_gather_top_experts_buffer.data_ptr()
        if all_gather_top_experts_buffer_handle is None
        else all_gather_top_experts_buffer_handle.multicast_ptr
    )

    barrier_buffer, barrier_buffer_handle, barrier_buffer_ptrs = allocate_buffer(
        1, dtype=torch.int32, zero=True
    )
    barrier_buffer_multicast_ptr = int(
        barrier_buffer.data_ptr()
        if barrier_buffer_handle is None
        else barrier_buffer_handle.multicast_ptr
    )
    barrier_target = torch.zeros(1, dtype=torch.int32, device=device)

    dist.barrier(
        group=group, async_op=True, device_ids=[device_index]
    ).block_current_stream()

    workspace = MoKWorkspace(
        group_name=group_name,
        ep_rank=ep_rank,
        ep_size=ep_size,
        device=device,
        num_local_tokens=num_local_tokens,
        hidden_size=hidden_size,
        topk=topk,
        schedule_capacity=schedule_capacity,
        x_buffer=x_buffer,
        x_buffer_handle=x_buffer_handle,
        x_buffer_ptrs=x_buffer_ptrs,
        combine_buffer=combine_buffer,
        combine_buffer_handle=combine_buffer_handle,
        combine_buffer_ptrs=combine_buffer_ptrs,
        d_y_buffer=d_y_buffer,
        d_y_buffer_handle=d_y_buffer_handle,
        d_y_buffer_ptrs=d_y_buffer_ptrs,
        d_x_routed_buffer=d_x_routed_buffer,
        d_x_routed_buffer_handle=d_x_routed_buffer_handle,
        d_x_routed_buffer_ptrs=d_x_routed_buffer_ptrs,
        router_weight_buffer=router_weight_buffer,
        router_weight_buffer_handle=router_weight_buffer_handle,
        router_weight_buffer_ptrs=router_weight_buffer_ptrs,
        d_router_weight_buffer=d_router_weight_buffer,
        d_router_weight_buffer_handle=d_router_weight_buffer_handle,
        d_router_weight_buffer_ptrs=d_router_weight_buffer_ptrs,
        all_gather_top_experts_buffer=all_gather_top_experts_buffer,
        all_gather_top_experts_buffer_handle=all_gather_top_experts_buffer_handle,
        all_gather_top_experts_buffer_multicast_ptr=all_gather_top_experts_buffer_multicast_ptr,
        barrier_buffer=barrier_buffer,
        barrier_buffer_handle=barrier_buffer_handle,
        barrier_buffer_ptrs=barrier_buffer_ptrs,
        barrier_buffer_multicast_ptr=barrier_buffer_multicast_ptr,
        barrier_target=barrier_target,
    )
    return workspace


def validate_inputs(
    config: MoKConfig,
    workspace: MoKWorkspace,
    schedule: MoKSchedule,
    x: torch.Tensor,
    router_weights: torch.Tensor | None = None,
    grad_output: torch.Tensor | None = None,
) -> None:
    """Validates runtime inputs against the workspace and schedule.

    Inputs:
        config:         MoKConfig
        workspace:      MoKWorkspace
        schedule:       MoKSchedule
        x:              bfloat16 [num_local_tokens, hidden_size]
        router_weights: float32 [num_local_tokens, topk] | None
        grad_output:    bfloat16 [num_local_tokens, hidden_size] | None

    Outputs:
        None
    """
    if not isinstance(config, MoKConfig):
        raise TypeError("config must be a MoKConfig")
    if not isinstance(workspace, MoKWorkspace):
        raise TypeError("workspace must be a MoKWorkspace")
    if not isinstance(schedule, MoKSchedule):
        raise TypeError("schedule must be a MoKSchedule")
    expected_activation_shape = (workspace.num_local_tokens, workspace.hidden_size)
    tensors = [("x", x, torch.bfloat16, expected_activation_shape)]
    if grad_output is not None:
        tensors.append(
            ("grad_output", grad_output, torch.bfloat16, expected_activation_shape)
        )
    if router_weights is not None:
        tensors.append(
            (
                "router_weights",
                router_weights,
                torch.float32,
                (workspace.num_local_tokens, workspace.topk),
            )
        )
    for tensor_name, tensor, expected_dtype, expected_shape in tensors:
        if (
            not tensor.is_cuda
            or tensor.device != workspace.device
            or tensor.dtype != expected_dtype
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                f"{tensor_name} must be contiguous {expected_dtype} on the workspace CUDA device"
            )
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{tensor_name} shape does not match the workspace")
    if schedule.peer_rank.numel() != workspace.schedule_capacity:
        raise ValueError("schedule capacity does not match the workspace")
