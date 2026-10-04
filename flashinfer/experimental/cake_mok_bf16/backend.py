# Copyright 2026 Cursor Research
# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# Modified: standalone CUDA module loading; original host launch boundaries.


from dataclasses import dataclass, replace
import math

import torch
import torch.distributed as dist

from . import workspace as native
from ._kernels import (
    MoKCommunication,
    MoKScheduler,
    MoKForward,
    MoKBackward,
    MoKEpilogues,
)


@dataclass(frozen=True, slots=True)
class MoKSourceWorkspace:
    """Caller-owned physical storage; logical lengths belong to each schedule.

    ``source_capacity`` is common across ranks, not the current input length.
    Keep this object alive while any captured graph or context uses its buffers.
    """

    storage: native.MoKWorkspace
    initial_source_counts: tuple[int, ...]

    @property
    def source_capacity(self):
        return self.storage.num_local_tokens


@dataclass(frozen=True, slots=True)
class MoKSourceSchedule(native.MoKSchedule):
    num_source_tokens: int
    workspace: MoKSourceWorkspace


@dataclass(frozen=True, slots=True)
class MoKSourceForwardContext(native.MoKForwardContext):
    schedule: MoKSourceSchedule


class MoKFunctional:
    """Precompile kernels before capture; retain native function signatures."""

    def __init__(self, ep, local_experts, topk):
        if (ep, local_experts, topk) not in ((1, 4, 2), (4, 4, 2), (16, 16, 8)):
            raise ValueError("Unsupported exported (EP, local experts, top-k) layout")
        self.ep, self.local_experts, self.topk = ep, local_experts, topk
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.communication = MoKCommunication()
        self.scheduler = MoKScheduler(ep, local_experts)
        self.forward_kernel = MoKForward()
        self.backward_kernel = MoKBackward()
        self.epilogues = MoKEpilogues(topk)

    def create_workspace(
        self,
        config,
        group,
        *,
        device,
        num_local_tokens,
        hidden_size,
        topk,
        source_capacity=None,
    ):
        """Collectively reserve storage for rank-local input lengths."""
        if dist.get_world_size(group) != self.ep or topk != self.topk:
            raise ValueError("Workspace differs from the prepared EP/top-k geometry")
        return create_source_workspace(
            config,
            group,
            device=device,
            num_local_tokens=num_local_tokens,
            hidden_size=hidden_size,
            topk=topk,
            source_capacity=source_capacity,
        )

    @staticmethod
    def _inputs(config, workspace, schedule, x, router_weights, grad_output=None):
        if isinstance(workspace, MoKSourceWorkspace):
            if (
                not isinstance(schedule, MoKSourceSchedule)
                or schedule.workspace is not workspace
            ):
                raise ValueError(
                    "A variable-source schedule must belong to this workspace"
                )
            count = schedule.num_source_tokens
            storage = workspace.storage
            if not 0 <= count <= workspace.source_capacity:
                raise ValueError("Schedule source count exceeds workspace capacity")
            # A metadata-only view reuses the native dtype/device/shape checks.
            # No buffer is allocated or resized and the shared workspace is not mutated.
            logical = replace(storage, num_local_tokens=count)
            native.validate_inputs(
                config, logical, schedule, x, router_weights, grad_output
            )
            return storage, count
        if isinstance(schedule, MoKSourceSchedule):
            raise ValueError(
                "A variable-source schedule requires its owning workspace wrapper"
            )
        native.validate_inputs(
            config, workspace, schedule, x, router_weights, grad_output
        )
        return workspace, workspace.num_local_tokens

    @staticmethod
    def _copy_source(destination, source, padding):
        count = source.shape[0]
        if count == destination.shape[0]:
            destination.copy_(source)
        else:
            destination[:count].copy_(source)
            destination[count:].fill_(padding)

    def _check_device(self, workspace):
        if (
            workspace.device != self.device
            or torch.cuda.current_device() != self.device.index
        ):
            raise ValueError(
                "Use the prepared CUDA device for this workspace and adapter"
            )

    def _barrier(self, workspace):
        self.communication.barrier_all(
            workspace.barrier_buffer,
            workspace.barrier_buffer_ptrs,
            workspace.barrier_buffer_multicast_ptr,
            workspace.barrier_target,
        )

    def build_schedule(self, workspace, config, top_experts, *, num_local_experts):
        owner = workspace if isinstance(workspace, MoKSourceWorkspace) else None
        if owner is not None:
            workspace = owner.storage
        if not isinstance(workspace, native.MoKWorkspace) or not isinstance(
            config, native.MoKConfig
        ):
            raise TypeError("Native MoK workspace/config objects are required")
        if (
            workspace.ep_size != self.ep
            or workspace.topk != self.topk
            or num_local_experts != self.local_experts
        ):
            raise ValueError(
                "Workspace/expert geometry differs from the compiled scheduler"
            )
        self._check_device(workspace)
        properties = torch.cuda.get_device_properties(workspace.device)
        for count in (config.fwd_num_comm_sms, config.bwd_num_comm_sms):
            if (
                type(count) is not int
                or count <= 0
                or count % 2
                or count >= properties.multi_processor_count
            ):
                raise ValueError(
                    "Communication-SM counts must be positive, even and leave compute SMs"
                )
        if (
            type(config.minibatch_size) is not int
            or config.minibatch_size <= 0
            or config.minibatch_size % 256
            or type(config.macrobatch_size) is not int
            or config.macrobatch_size <= 0
            or config.macrobatch_size % config.minibatch_size
        ):
            raise ValueError("Native minibatch/macrobatch geometry is required")
        if config.all_gather_top_experts_chunk_bytes != 2048:
            raise ValueError(
                "This Cake metadata pipeline currently uses the native 2048-byte chunk"
            )
        if not isinstance(top_experts, torch.Tensor) or top_experts.ndim != 2:
            raise ValueError("Expert IDs must have shape [source_tokens, topk]")
        count = top_experts.shape[0]
        if owner is not None and count > owner.source_capacity:
            raise ValueError(
                "Input length exceeds source capacity; collectively recreate the workspace"
            )
        expected_count = count if owner is not None else workspace.num_local_tokens
        if (
            top_experts.device != workspace.device
            or top_experts.dtype != torch.int64
            or not top_experts.is_contiguous()
            or tuple(top_experts.shape) != (expected_count, workspace.topk)
        ):
            raise ValueError(
                "Expert IDs must be contiguous int64 on the workspace device with its source shape"
            )
        if workspace.num_local_tokens * workspace.topk * 4 % 2048:
            raise ValueError("The route buffer must contain complete metadata chunks")
        if count == workspace.num_local_tokens:
            top_experts_int32 = top_experts.to(torch.int32)
        else:
            top_experts_int32 = torch.full(
                (workspace.num_local_tokens, workspace.topk),
                -1,
                dtype=torch.int32,
                device=workspace.device,
            )
            top_experts_int32[:count].copy_(top_experts)
        self.communication.all_gather_top_experts(
            top_experts_int32,
            workspace.all_gather_top_experts_buffer,
            workspace.all_gather_top_experts_buffer_multicast_ptr,
            workspace.ep_rank,
        )
        self._barrier(workspace)
        rank, token, count, counts = self.scheduler(
            workspace.all_gather_top_experts_buffer,
            workspace.schedule_capacity,
            workspace.ep_rank,
        )
        if owner is not None:
            return MoKSourceSchedule(
                peer_rank=rank,
                peer_token_idx=token,
                num_tokens=count,
                tokens_per_expert=counts,
                num_source_tokens=top_experts.shape[0],
                workspace=owner,
            )
        return native.MoKSchedule(
            peer_rank=rank,
            peer_token_idx=token,
            num_tokens=count,
            tokens_per_expert=counts,
        )

    @staticmethod
    def _schedule(schedule):
        return (
            schedule.peer_rank,
            schedule.peer_token_idx,
            schedule.num_tokens,
            schedule.tokens_per_expert,
        )

    def _weights(self, x, *weights):
        if any(
            not isinstance(weight, torch.Tensor)
            or weight.dtype != torch.bfloat16
            or weight.device != x.device
            or not weight.is_contiguous()
            for weight in weights
        ):
            raise ValueError(
                "The Cake training path requires contiguous CUDA BF16 expert weights"
            )
        if weights[0].ndim != 2:
            raise ValueError("Shared gate weights must be a matrix")
        intermediate, hidden = weights[0].shape
        shapes = (
            (intermediate, hidden),
            (intermediate, hidden),
            (hidden, intermediate),
            (self.local_experts, intermediate, hidden),
            (self.local_experts, intermediate, hidden),
            (self.local_experts, hidden, intermediate),
        )
        if (
            hidden != x.shape[1]
            or intermediate <= 0
            or intermediate % 256
            or any(
                tuple(w.shape) != shape
                for w, shape in zip(weights, shapes, strict=True)
            )
        ):
            raise ValueError(
                "Expert weight shapes must match the prepared workspace and 256-column tiles"
            )

    def forward(
        self,
        config,
        workspace,
        schedule,
        x,
        router_weights,
        shared_gate_weights,
        shared_up_weights,
        shared_down_weights,
        routed_gate_weights,
        routed_up_weights,
        routed_down_weights,
        swiglu_limit=None,
    ):
        workspace, source_count = self._inputs(
            config, workspace, schedule, x, router_weights
        )
        self._check_device(workspace)
        self._weights(
            x,
            shared_gate_weights,
            shared_up_weights,
            shared_down_weights,
            routed_gate_weights,
            routed_up_weights,
            routed_down_weights,
        )
        self._copy_source(workspace.x_buffer, x, 0)
        self._copy_source(workspace.router_weight_buffer, router_weights, 1)
        if source_count < workspace.num_local_tokens:
            workspace.combine_buffer[source_count * workspace.topk :].zero_()
        self._barrier(workspace)
        values = self.forward_kernel(
            workspace.x_buffer,
            workspace.x_buffer_ptrs,
            workspace.combine_buffer,
            workspace.combine_buffer_ptrs,
            shared_gate_weights,
            routed_gate_weights,
            shared_up_weights,
            routed_up_weights,
            shared_down_weights,
            routed_down_weights,
            *self._schedule(schedule),
            workspace.topk,
            swiglu_limit,
            config.fwd_num_comm_sms,
            config.macrobatch_size,
            config.minibatch_size,
        )
        context_type = (
            MoKSourceForwardContext
            if isinstance(schedule, MoKSourceSchedule)
            else native.MoKForwardContext
        )
        context_metadata = (
            {"schedule": schedule} if isinstance(schedule, MoKSourceSchedule) else {}
        )
        context = context_type(
            x_routed=values[0],
            gate_shared=values[1],
            gate_routed=values[2],
            up_shared=values[3],
            up_routed=values[4],
            hidden_shared=values[5],
            hidden_routed=values[6],
            **context_metadata,
        )
        self._barrier(workspace)
        output = self.epilogues.forward(
            values[7], workspace.combine_buffer, workspace.router_weight_buffer
        )
        return output[:source_count], context

    def backward(
        self,
        config,
        workspace,
        schedule,
        forward_context,
        grad_output,
        x,
        router_weights,
        shared_gate_weights,
        shared_up_weights,
        shared_down_weights,
        routed_gate_weights,
        routed_up_weights,
        routed_down_weights,
        swiglu_limit=None,
    ):
        workspace, source_count = self._inputs(
            config, workspace, schedule, x, router_weights, grad_output
        )
        if isinstance(schedule, MoKSourceSchedule):
            if (
                not isinstance(forward_context, MoKSourceForwardContext)
                or forward_context.schedule is not schedule
            ):
                raise ValueError(
                    "Backward requires the matching variable-source forward context"
                )
        elif isinstance(forward_context, MoKSourceForwardContext):
            raise ValueError("A variable-source context requires its matching schedule")
        if not isinstance(forward_context, native.MoKForwardContext):
            raise TypeError("A native MoKForwardContext is required")
        self._check_device(workspace)
        self._weights(
            x,
            shared_gate_weights,
            shared_up_weights,
            shared_down_weights,
            routed_gate_weights,
            routed_up_weights,
            routed_down_weights,
        )
        self._copy_source(workspace.d_y_buffer, grad_output, 0)
        self._copy_source(workspace.x_buffer, x, 0)
        self._copy_source(workspace.router_weight_buffer, router_weights, 1)
        if source_count < workspace.num_local_tokens:
            workspace.d_x_routed_buffer[source_count * workspace.topk :].zero_()
            workspace.d_router_weight_buffer[source_count:].zero_()
        self._barrier(workspace)
        values = self.backward_kernel(
            workspace.d_y_buffer,
            workspace.d_y_buffer_ptrs,
            workspace.d_x_routed_buffer,
            workspace.d_x_routed_buffer_ptrs,
            workspace.router_weight_buffer,
            workspace.router_weight_buffer_ptrs,
            workspace.d_router_weight_buffer,
            workspace.d_router_weight_buffer_ptrs,
            shared_gate_weights,
            routed_gate_weights,
            shared_up_weights,
            routed_up_weights,
            shared_down_weights,
            routed_down_weights,
            forward_context.x_routed,
            forward_context.gate_shared,
            forward_context.gate_routed,
            forward_context.up_shared,
            forward_context.up_routed,
            forward_context.hidden_shared,
            forward_context.hidden_routed,
            workspace.x_buffer,
            workspace.x_buffer_ptrs,
            *self._schedule(schedule),
            workspace.topk,
            swiglu_limit,
            config.bwd_num_comm_sms,
            config.macrobatch_size,
            config.minibatch_size,
        )
        self._barrier(workspace)
        dx = self.epilogues.backward(values[0], workspace.d_x_routed_buffer)
        dscores = workspace.d_router_weight_buffer[:source_count].clone()
        return (
            dx[:source_count],
            dscores,
            values[10],
            values[12],
            values[14],
            values[9],
            values[11],
            values[13],
        )


def create_source_workspace(
    config, group, *, device, num_local_tokens, hidden_size, topk, source_capacity=None
):
    """Collectively reserve storage for potentially unequal source counts.

    Setup only: negotiate counts/geometry once, then allocate native symmetric
    buffers with a common aligned capacity. Counts may change within that
    capacity without another collective. Shape/address changes require graph
    recapture; unchanged-shape input updates may replay the existing graph.
    ``source_capacity`` reserves future growth and must agree across ranks.
    Received-route capacity remains controlled by the native config multiplier.
    """
    if not isinstance(config, native.MoKConfig):
        raise TypeError("A native MoKConfig is required")
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "Workspace creation is collective setup, outside CUDA Graph capture"
        )
    ep = dist.get_world_size(group)
    gathered = [None] * ep
    dist.all_gather_object(
        gathered,
        (num_local_tokens, hidden_size, topk, source_capacity, config),
        group=group,
    )
    counts = tuple(row[0] for row in gathered)
    if any(type(n) is not int or n < 0 for n in counts):
        raise ValueError("Source token counts must be nonnegative integers")
    if any(row[1:] != gathered[0][1:] for row in gathered):
        raise ValueError("Ranks must agree on hidden size, top-k, capacity and config")
    if type(topk) is not int or not 0 < topk <= 255:
        raise ValueError("Top-k must be an integer in [1, 255]")
    if source_capacity is not None and (
        type(source_capacity) is not int or source_capacity < max(counts)
    ):
        raise ValueError(
            "source_capacity must be an integer covering every source count"
        )
    # Native compute tiles need 256 rows; metadata uses 2048-byte int32 chunks.
    alignment = math.lcm(256, 512 // math.gcd(topk, 512))
    requested = max(counts) if source_capacity is None else source_capacity
    capacity = ((max(512, requested) + alignment - 1) // alignment) * alignment
    storage = native.create_workspace(
        config,
        group,
        device=device,
        num_local_tokens=capacity,
        hidden_size=hidden_size,
        topk=topk,
    )
    return MoKSourceWorkspace(storage, counts)
