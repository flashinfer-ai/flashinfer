# Copyright 2026 Cursor Research
# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# Modified: standalone CUDA module loading; original host launch boundaries.


import torch
from . import workspace as native
from ._kernels import (
    MoKCommunication,
    MoKScheduler,
    MoKForward,
    MoKBackward,
    MoKEpilogues,
)


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
        if not isinstance(workspace, native.MoKWorkspace) or not isinstance(
            config, native.MoKConfig
        ):
            raise TypeError("MoK workspace/config objects are required")
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
        if (
            top_experts.device != workspace.device
            or top_experts.dtype != torch.int64
            or not top_experts.is_contiguous()
            or tuple(top_experts.shape) != (workspace.num_local_tokens, workspace.topk)
        ):
            raise ValueError(
                "Expert IDs must be contiguous int64 on the workspace device with its source shape"
            )
        if workspace.num_local_tokens * workspace.topk * 4 % 2048:
            raise ValueError("The route buffer must contain complete metadata chunks")
        top_experts_int32 = top_experts.to(torch.int32)
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
        self._check_device(workspace)
        native.validate_inputs(config, workspace, schedule, x, router_weights)
        self._weights(
            x,
            shared_gate_weights,
            shared_up_weights,
            shared_down_weights,
            routed_gate_weights,
            routed_up_weights,
            routed_down_weights,
        )
        workspace.x_buffer.copy_(x)
        workspace.router_weight_buffer.copy_(router_weights)
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
        context = native.MoKForwardContext(
            x_routed=values[0],
            gate_shared=values[1],
            gate_routed=values[2],
            up_shared=values[3],
            up_routed=values[4],
            hidden_shared=values[5],
            hidden_routed=values[6],
        )
        self._barrier(workspace)
        output = self.epilogues.forward(
            values[7], workspace.combine_buffer, workspace.router_weight_buffer
        )
        return output, context

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
        self._check_device(workspace)
        native.validate_inputs(
            config, workspace, schedule, x, router_weights, grad_output
        )
        if not isinstance(forward_context, native.MoKForwardContext):
            raise TypeError("A MoKForwardContext is required")
        self._weights(
            x,
            shared_gate_weights,
            shared_up_weights,
            shared_down_weights,
            routed_gate_weights,
            routed_up_weights,
            routed_down_weights,
        )
        workspace.d_y_buffer.copy_(grad_output)
        workspace.x_buffer.copy_(x)
        workspace.router_weight_buffer.copy_(router_weights)
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
        dscores = workspace.d_router_weight_buffer.clone()
        return (
            dx,
            dscores,
            values[10],
            values[12],
            values[14],
            values[9],
            values[11],
            values[13],
        )
