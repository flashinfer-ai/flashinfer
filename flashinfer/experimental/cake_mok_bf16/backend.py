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
    MoKForwardMxfp8,
    MoKBackwardMxfp8,
    MoKMxfp8Quantize,
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
class MoKForwardContext(native.MoKForwardContext):
    schedule: native.MoKSchedule


def context_defined_rows(config, context):
    """Rows of a forward or recomputed context that the kernels define.

    The routed rings (``x_routed``/``gate_routed``/``up_routed``/``hidden_routed``)
    hold the retained macrobatch, i.e. their first ``min(macrobatch_size, routed
    rows)`` rows; the shared activations cover the real source rows. Rows beyond
    these are never written and hold whatever the allocator handed out, so
    bitwise comparisons (saved vs. recomputed context) must stop there.
    Synchronizes on the schedule's routed row count; not for graph capture."""
    schedule = context.schedule
    routed = min(config.macrobatch_size, int(schedule.num_tokens.item()))
    return routed, schedule.num_source_tokens


# (EP, local experts, top-k) layouts with exported scheduler kernels: the toy
# layouts of the tests, GLM-5.2 (256 routed experts) and GLM-5.3-Flash (288).
SUPPORTED_LAYOUTS = (
    (1, 4, 2),
    (4, 4, 2),
    (16, 16, 8),
    (64, 4, 8),
    (4, 64, 8),
    (8, 32, 8),
    (32, 8, 8),
    (8, 36, 8),
    (32, 9, 8),
)


_MXFP8_KERNELS = {
    "forward": MoKForwardMxfp8,
    "backward": MoKBackwardMxfp8,
    "backward_f32": lambda: MoKBackwardMxfp8(wgrad_f32=True),
}
_QUANTIZERS = {}


def mxfp8_quantize(x_bf16, return_normal=True, return_transposed=True):
    """MoK's ``ops.mxfp8_quantize`` (E4M3 data, E8M0 scales per 32-element K block).

    Returns ``(x_fp8, x_sc, x_fp8_t, x_sc_t)`` with ``None`` for layouts not
    requested: ``[E, M, N]`` E4M3 data with ``[E * M / 128, N / 128, 32, 16]``
    scale tiles, and the transposed ``[E, N, M]`` / ``[E * N / 128, M / 128, 32, 16]``
    pair. A 2-D input returns 2-D data. Compiles on first use per device
    (outside CUDA Graph capture).
    """
    if not isinstance(x_bf16, torch.Tensor) or not x_bf16.is_cuda:
        raise ValueError("x_bf16 must be a CUDA tensor")
    key = x_bf16.device.index
    if key not in _QUANTIZERS:
        with torch.cuda.device(x_bf16.device):
            _QUANTIZERS[key] = MoKMxfp8Quantize()
    return _QUANTIZERS[key](x_bf16, return_normal, return_transposed)


class MoKFunctional:
    """Precompile kernels before capture; retain native function signatures.

    Routed expert weights follow MoK's conventions: BF16 tensors select the
    BF16 kernels; MXFP8 tensor tuples from :func:`mxfp8_quantize` select the
    native MXFP8 kernels (``forward``/``recompute_forward_context`` take the
    ``(w_fp8, w_sc)`` pairs, ``backward`` the gate/up 4-tuples and the down
    ``(w_t_fp8, w_t_sc)`` pair). ``mxfp8=True`` precompiles the MXFP8 kernels;
    otherwise they load on first use, outside CUDA Graph capture.
    """

    def __init__(self, ep, local_experts, topk, mxfp8=False):
        if (ep, local_experts, topk) not in SUPPORTED_LAYOUTS:
            raise ValueError("Unsupported exported (EP, local experts, top-k) layout")
        self.ep, self.local_experts, self.topk = ep, local_experts, topk
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.communication = MoKCommunication()
        self.scheduler = MoKScheduler(ep, local_experts)
        self.forward_kernel = MoKForward()
        self.backward_kernel = MoKBackward()
        self.epilogues = MoKEpilogues(topk)
        self._mxfp8 = {}
        if mxfp8:
            for role in ("forward", "backward"):
                self._mxfp8_kernel(role)

    def _mxfp8_kernel(self, role):
        if role not in self._mxfp8:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Prepare the MXFP8 MoK kernels before CUDA Graph capture "
                    "(prepare_mok_bf16(..., mxfp8=True) or one eager call)"
                )
            self._mxfp8[role] = _MXFP8_KERNELS[role]()
        return self._mxfp8[role]

    @staticmethod
    def _is_mxfp8(*routed_weights):
        """MoK's signature: MXFP8 routed weights arrive as tensor tuples."""
        flags = [isinstance(w, (tuple, list)) for w in routed_weights]
        if any(flags) and not all(flags):
            raise ValueError(
                "Routed expert weights must all be BF16 tensors or all MXFP8 tuples"
            )
        return all(flags)

    def _mxfp8_weights(self, x, shared_weights, routed_weights):
        """Shared BF16 weights plus MXFP8 tuples; the kernels validate the exact
        tuple arity and the E4M3/scale-tile shapes."""
        self._weights(x, *shared_weights, None, None, None)
        for weights in routed_weights:
            for member in weights:
                if member is None:
                    continue
                if (
                    not isinstance(member, torch.Tensor)
                    or member.device != x.device
                    or not member.is_contiguous()
                    or member.dtype not in (torch.float8_e4m3fn, torch.uint8)
                ):
                    raise ValueError(
                        "MXFP8 routed weights must be contiguous CUDA E4M3 data and "
                        "uint8 scale tiles from mxfp8_quantize"
                    )

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
    def _copy_source(destination, source):
        # Only the real rows are staged; peers read routed rows through the
        # schedule and the fused kernels bound shared work by the row count.
        count = source.shape[0]
        if count:
            destination[:count].copy_(source)

    @staticmethod
    def _rows(buffer, count):
        """A view of the real rows, or a one-row TMA placeholder when empty."""
        return buffer[: max(count, 1)]

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
        """BF16 weight checks; ``None`` routed entries are validated by the MXFP8 path."""
        if any(
            not isinstance(weight, torch.Tensor)
            or weight.dtype != torch.bfloat16
            or weight.device != x.device
            or not weight.is_contiguous()
            for weight in weights
            if weight is not None
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
                w is not None and tuple(w.shape) != shape
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
        shared = (shared_gate_weights, shared_up_weights, shared_down_weights)
        routed = (routed_gate_weights, routed_up_weights, routed_down_weights)
        mxfp8 = self._is_mxfp8(*routed)
        if mxfp8:
            self._mxfp8_weights(x, shared, routed)
        else:
            self._weights(x, *shared, *routed)
        self._copy_source(workspace.x_buffer, x)
        self._copy_source(workspace.router_weight_buffer, router_weights)
        self._barrier(workspace)
        kernel = self._mxfp8_kernel("forward") if mxfp8 else self.forward_kernel
        values = kernel(
            self._rows(workspace.x_buffer, source_count),
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
            source_rows=source_count,
        )
        context = self._context(values, schedule, mxfp8)
        self._barrier(workspace)
        if source_count == 0:
            return x.new_empty((0, workspace.hidden_size)), context
        output = self.epilogues.forward(
            values[11 if mxfp8 else 7][:source_count],
            workspace.combine_buffer[: source_count * workspace.topk],
            workspace.router_weight_buffer[:source_count],
        )
        return output, context

    @staticmethod
    def _context(values, schedule, mxfp8):
        """MoK's context: BF16 rings, or MXFP8 ``(E4M3, scale tiles)`` pairs (``x`` and
        ``hidden`` transposed, ``gate``/``up`` normal) as the native MXFP8 functional layer."""
        if mxfp8:
            return MoKForwardContext(
                x_routed=(values[0], values[1]),
                gate_shared=values[2],
                gate_routed=(values[3], values[4]),
                up_shared=values[5],
                up_routed=(values[6], values[7]),
                hidden_shared=values[8],
                hidden_routed=(values[9], values[10]),
                schedule=schedule,
            )
        return MoKForwardContext(
            x_routed=values[0],
            gate_shared=values[1],
            gate_routed=values[2],
            up_shared=values[3],
            up_routed=values[4],
            hidden_shared=values[5],
            hidden_routed=values[6],
            schedule=schedule,
        )

    def recompute_forward_context(
        self,
        config,
        workspace,
        schedule,
        x,
        shared_gate_weights,
        shared_up_weights,
        routed_gate_weights,
        routed_up_weights,
        swiglu_limit=None,
    ):
        """Rebuild the backward context from ``x`` (MoK ``recompute_forward_context``).

        Runs dispatch, both gate/up expert GEMMs and the SwiGLU only: no down
        projections, no combine, no output. The result is bitwise identical to
        the context ``forward`` saved for the same inputs and schedule, so a
        backward from it reproduces the saved-context backward bitwise.
        """
        workspace, source_count = self._inputs(config, workspace, schedule, x, None)
        self._check_device(workspace)
        mxfp8 = self._is_mxfp8(routed_gate_weights, routed_up_weights)
        # The unused down projections are absent from the shape checks: no
        # placeholder tensors are materialized per call (a transposed copy of
        # the routed gate weights cost ~4 ms per recompute at the GLM-5.2 shape).
        if mxfp8:
            self._mxfp8_weights(
                x,
                (shared_gate_weights, shared_up_weights, None),
                (routed_gate_weights, routed_up_weights),
            )
            # The MXFP8 kernel takes no down weights in recompute mode.
            routed_down_weights = None
        else:
            self._weights(
                x,
                shared_gate_weights,
                shared_up_weights,
                None,
                routed_gate_weights,
                routed_up_weights,
                None,
            )
            # The BF16 kernel's down-weight TMA maps need a tensor; unused.
            routed_down_weights = routed_gate_weights
        self._copy_source(workspace.x_buffer, x)
        self._barrier(workspace)
        # The down weights are unused placeholders in recompute mode.
        kernel = self._mxfp8_kernel("forward") if mxfp8 else self.forward_kernel
        values = kernel(
            self._rows(workspace.x_buffer, source_count),
            workspace.x_buffer_ptrs,
            workspace.combine_buffer,
            workspace.combine_buffer_ptrs,
            shared_gate_weights,
            routed_gate_weights,
            shared_up_weights,
            routed_up_weights,
            shared_gate_weights,
            routed_down_weights,
            *self._schedule(schedule),
            workspace.topk,
            swiglu_limit,
            config.fwd_num_comm_sms,
            config.macrobatch_size,
            config.minibatch_size,
            source_rows=source_count,
            recompute_only=True,
        )
        context = self._context(values, schedule, mxfp8)
        self._barrier(workspace)
        return context

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
        wgrad_f32=False,
    ):
        """MoK ``backward``; ``wgrad_f32`` (MXFP8 only) returns the routed weight
        gradients in FP32 with FP32 accumulation across macrobatches."""
        workspace, source_count = self._inputs(
            config, workspace, schedule, x, router_weights, grad_output
        )
        if (
            not isinstance(forward_context, MoKForwardContext)
            or forward_context.schedule is not schedule
        ):
            raise ValueError(
                "Backward requires the matching forward context and schedule"
            )
        self._check_device(workspace)
        shared = (shared_gate_weights, shared_up_weights, shared_down_weights)
        routed = (routed_gate_weights, routed_up_weights, routed_down_weights)
        mxfp8 = self._is_mxfp8(*routed)
        if mxfp8 != isinstance(forward_context.x_routed, tuple):
            raise ValueError(
                "The forward context precision must match the routed weights"
            )
        if wgrad_f32 and not mxfp8:
            raise ValueError("wgrad_f32 requires MXFP8 routed weights")
        if mxfp8:
            self._mxfp8_weights(x, shared, routed)
        else:
            self._weights(x, *shared, *routed)
        self._copy_source(workspace.d_y_buffer, grad_output)
        self._copy_source(workspace.x_buffer, x)
        self._copy_source(workspace.router_weight_buffer, router_weights)
        self._barrier(workspace)
        if mxfp8:
            return self._backward_mxfp8(
                config,
                workspace,
                schedule,
                forward_context,
                x,
                source_count,
                shared,
                routed,
                swiglu_limit,
                wgrad_f32,
            )
        values = self.backward_kernel(
            self._rows(workspace.d_y_buffer, source_count),
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
            self._rows(workspace.x_buffer, source_count),
            workspace.x_buffer_ptrs,
            *self._schedule(schedule),
            workspace.topk,
            swiglu_limit,
            config.bwd_num_comm_sms,
            config.macrobatch_size,
            config.minibatch_size,
            source_rows=source_count,
        )
        self._barrier(workspace)
        if source_count == 0:
            dx = x.new_empty((0, workspace.hidden_size))
        else:
            dx = self.epilogues.backward(
                values[0][:source_count],
                workspace.d_x_routed_buffer[: source_count * workspace.topk],
            )
        dscores = workspace.d_router_weight_buffer[:source_count].clone()
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

    def _backward_mxfp8(
        self,
        config,
        workspace,
        schedule,
        forward_context,
        x,
        source_count,
        shared,
        routed,
        swiglu_limit,
        wgrad_f32,
    ):
        kernel = self._mxfp8_kernel("backward_f32" if wgrad_f32 else "backward")
        values = kernel(
            self._rows(workspace.d_y_buffer, source_count),
            workspace.d_y_buffer_ptrs,
            workspace.d_x_routed_buffer,
            workspace.d_x_routed_buffer_ptrs,
            workspace.router_weight_buffer,
            workspace.router_weight_buffer_ptrs,
            workspace.d_router_weight_buffer,
            workspace.d_router_weight_buffer_ptrs,
            shared[0],
            routed[0],
            shared[1],
            routed[1],
            shared[2],
            routed[2],
            *forward_context.x_routed,
            forward_context.gate_shared,
            *forward_context.gate_routed,
            forward_context.up_shared,
            *forward_context.up_routed,
            forward_context.hidden_shared,
            *forward_context.hidden_routed,
            self._rows(workspace.x_buffer, source_count),
            workspace.x_buffer_ptrs,
            *self._schedule(schedule),
            workspace.topk,
            swiglu_limit,
            config.bwd_num_comm_sms,
            config.macrobatch_size,
            config.minibatch_size,
            source_rows=source_count,
        )
        self._barrier(workspace)
        if source_count == 0:
            dx = x.new_empty((0, workspace.hidden_size))
        else:
            dx = self.epilogues.backward(
                values[0][:source_count],
                workspace.d_x_routed_buffer[: source_count * workspace.topk],
            )
        dscores = workspace.d_router_weight_buffer[:source_count].clone()
        # MoK's 18-output order: routed gradients at 13/15/17, shared at 12/14/16.
        return (
            dx,
            dscores,
            values[13],
            values[15],
            values[17],
            values[12],
            values[14],
            values[16],
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
