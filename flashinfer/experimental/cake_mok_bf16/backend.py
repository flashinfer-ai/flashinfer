# Copyright 2026 Cursor Research
# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# Modified: standalone CUDA module loading; original host launch boundaries.
"""MoK functional adapter over the generated training kernels.

Copies, barriers, scheduler, epilogues and empty-expert zeroing retain the
native launch boundaries. Unequal source lengths use masked rows in common
symmetric storage; the fused kernels keep the same runtime-shape entrypoints.
Routed experts run in BF16, or natively in MXFP8 when the routed weights are
passed as caller-prequantized MXFP8 tuples, as in MoK's functional API.
"""

from dataclasses import dataclass, replace
import math
import os
import sys

import torch
import torch.distributed as dist

from . import workspace as native
from ._kernels import (
    EPILOGUE_TOPKS,
    SCHEDULER_LAYOUTS,
    MoKBackward,
    MoKBackwardMXFP8,
    MoKCommunication,
    MoKEpilogues,
    MoKForward,
    MoKForwardMXFP8,
    MoKRecompute,
    MoKRecomputeMXFP8,
    MoKScheduler,
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


@dataclass(frozen=True, slots=True, kw_only=True)
class MoKSourceSchedule(native.MoKSchedule):
    num_source_tokens: int
    workspace: MoKSourceWorkspace


@dataclass(frozen=True, slots=True, kw_only=True)
class MoKForwardContext(native.MoKForwardContext):
    schedule: native.MoKSchedule


def count_routed_load(routes, ep_size, local_experts):
    """Routed rows per owning rank over the gathered expert ids ``routes``.

    ``routes`` may have any shape; an id ``e`` in ``[0, ep_size * local_experts)``
    is owned by rank ``e // local_experts``. Negative ids (padding) and ids at or
    beyond the limit are not counted. Returns an int64 tensor of ``ep_size``
    counts on the device of ``routes``.

    One ``torch.histc`` pass with explicit bounds: no host synchronization
    (``torch.bincount`` sizes its output from a device ``max()``), no atomics
    on ``ep_size`` addresses (``scatter_add_``), and work and transient linear
    in the number of ids. The earlier compare-and-reduce form built an
    ``ep_size`` x ``ids`` boolean tensor and cost about 3 ms and 4.6 GB per call
    at EP64 (measured on GB200 by the FlashInfer reviewers of PR #6439). The
    owner ranks are histogrammed as float64 over ``[-1, ep_size + 1)`` in unit
    bins, so the integer bin arithmetic is exact: bin 0 collects the ``-1``
    padding, the last bin the ids at the limit, values outside the range are
    dropped, and the ``ep_size`` middle bins are the counts.
    """
    flat = routes.reshape(-1)
    owner = torch.div(flat, local_experts, rounding_mode="floor")
    hist = torch.histc(
        owner.to(torch.float64), bins=ep_size + 2, min=-1, max=ep_size + 1
    )
    return hist[1 : ep_size + 1].to(torch.int64)


def select_comm_split(loads, base, hot, threshold):
    """Communication-SM split for an observed per-rank routed-row ``loads`` sequence.

    Returns ``hot`` when it is set and the largest load is at least ``threshold``
    times the mean load (``sum(loads) / len(loads)``), else ``base``. Host-only;
    ``loads`` may be a list, a tuple or a CPU tensor. No observation yet or a
    zero total load (every rank empty) keeps ``base``.
    """
    if hot is None:
        return tuple(base)
    values = [int(value) for value in loads]
    total = sum(values)
    if total <= 0:
        return tuple(base)
    if max(values) * len(values) >= threshold * total:
        return tuple(hot)
    return tuple(base)


class MoKFunctional:
    """Precompile kernels before capture; retain native function signatures."""

    def __init__(
        self, ep, local_experts, topk, *, clamped_swiglu=False, fp32_wgrad=False
    ):
        """``clamped_swiglu`` selects which SwiGLU variant is compiled now.

        Plain (``swiglu_limit=None``) and clamped (finite positive limit)
        SwiGLU are separate kernel builds, as in native MoK's clamped
        template; the limit itself is a runtime value. Other variants (the
        other SwiGLU form, MXFP8 routed experts, context recompute) compile on
        first use, which must happen outside CUDA Graph capture; see
        :meth:`prepare`. ``fp32_wgrad`` selects backward kernels that add
        weight gradients into caller-owned FP32 accumulators
        (``backward(weight_grad_accumulators=...)``).
        """
        if (ep, local_experts) not in SCHEDULER_LAYOUTS or topk not in EPILOGUE_TOPKS:
            raise ValueError("Unsupported exported (EP, local experts, top-k) layout")
        if type(clamped_swiglu) is not bool or type(fp32_wgrad) is not bool:
            raise TypeError("clamped_swiglu and fp32_wgrad must be booleans")
        self.ep, self.local_experts, self.topk = ep, local_experts, topk
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.communication = MoKCommunication()
        self.scheduler = MoKScheduler(ep, local_experts)
        self.fp32_wgrad = fp32_wgrad
        self._forward_kernels, self._backward_kernels = {}, {}
        self._recompute_kernels = {}
        # id(native workspace) -> asynchronous per-rank routed-load observation.
        self._load_state = {}
        self._kernels(1.0 if clamped_swiglu else None)
        self.epilogues = MoKEpilogues(topk)

    def _kernels(self, swiglu_limit, mxfp8=False):
        """Kernel pair for (SwiGLU variant, routed precision); MXFP8 is native."""
        key = (mxfp8, swiglu_limit is not None)
        if key not in self._forward_kernels:
            if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Call prepare() for this variant before CUDA Graph capture"
                )
            forward, backward = (
                (MoKForwardMXFP8, MoKBackwardMXFP8)
                if mxfp8
                else (MoKForward, MoKBackward)
            )
            self._forward_kernels[key] = forward(key[1])
            self._backward_kernels[key] = backward(key[1], self.fp32_wgrad)
        return self._forward_kernels[key], self._backward_kernels[key]

    def _recompute_kernel(self, swiglu_limit, mxfp8=False):
        key = (mxfp8, swiglu_limit is not None)
        if key not in self._recompute_kernels:
            if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Call prepare_recompute() before CUDA Graph capture")
            recompute = MoKRecomputeMXFP8 if mxfp8 else MoKRecompute
            self._recompute_kernels[key] = recompute(key[1])
        return self._recompute_kernels[key]

    def prepare(self, swiglu_limit=None, mxfp8=False):
        """Compile the forward/backward kernels of this variant now."""
        self._kernels(swiglu_limit, mxfp8)

    def prepare_recompute(self, swiglu_limit=None, mxfp8=False):
        """Compile the context-recompute kernel for this variant now."""
        self._recompute_kernel(swiglu_limit, mxfp8)

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
    def _copy_source(destination, source, padding, rows):
        """Stage source rows; pad only the active tile prefix ``[count, rows)``.

        Shared-expert GEMMs, their weight gradients and the epilogues read
        exactly ``rows`` local rows; peers read only routed (valid) rows. Rows
        beyond the prefix are never read, so capacity does not add work.
        """
        count = source.shape[0]
        destination[:count].copy_(source)
        if rows > count:
            destination[count:rows].fill_(padding)

    @staticmethod
    def _shared_rows(source_count):
        # Peer addressing retains the symmetric allocation. Shared-expert work
        # only needs the active local prefix rounded to the existing MMA tile.
        return max(256, (source_count + 255) // 256 * 256)

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
        if config.imbalance_comm_sms is not None:
            hot = config.imbalance_comm_sms
            if (
                type(hot) is not tuple
                or len(hot) != 2
                or any(
                    type(count) is not int
                    or count <= 0
                    or count % 2
                    or count >= properties.multi_processor_count
                    for count in hot
                )
            ):
                raise ValueError(
                    "imbalance_comm_sms must be a (forward, backward) pair of positive, "
                    "even communication-SM counts that leave compute SMs"
                )
            threshold = config.imbalance_threshold
            if (
                type(threshold) not in (int, float)
                or not math.isfinite(threshold)
                or threshold <= 1.0
            ):
                raise ValueError(
                    "imbalance_threshold must be a finite number above 1.0"
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
        self._observe_load(workspace, config)
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

    def _load_entry(self, workspace, create):
        entry = self._load_state.get(id(workspace))
        if entry is None and create:
            entry = {
                "host": [
                    torch.empty(workspace.ep_size, dtype=torch.int64, pin_memory=True)
                    for _ in range(2)
                ],
                "events": [torch.cuda.Event() for _ in range(2)],
                "slot": 0,
                "pending": None,
                "loads": (),
                "split": None,
            }
            self._load_state[id(workspace)] = entry
        return entry

    def _observe_load(self, workspace, config):
        """Queue an asynchronous observation of the per-rank routed-row load.

        Every rank counts the gathered expert IDs per owning rank
        (``count_routed_load``: padding and invalid IDs excluded) on device and
        copies the counts into pinned host memory behind a CUDA event: no host
        synchronization. ``_comm_split`` reads the latest completed observation,
        so a split change takes effect from the first later launch whose event
        has completed. The gathered IDs, and so the counts, are identical on
        every rank, but each rank decides from its own latest completed
        observation: with back-to-back steps a launch may run with different
        splits on different ranks for a step. The split is a per-rank launch
        parameter (how many of that rank's SMs run its communication program);
        the protocol does not require the ranks to agree on it. Skipped during
        CUDA Graph capture: captured launches keep the split in force at
        capture time.
        """
        if (
            config.imbalance_comm_sms is None
            or torch.cuda.is_current_stream_capturing()
        ):
            return
        entry = self._load_entry(workspace, create=True)
        loads = count_routed_load(
            workspace.all_gather_top_experts_buffer,
            workspace.ep_size,
            self.local_experts,
        )
        slot = entry["slot"] ^ 1
        entry["host"][slot].copy_(loads, non_blocking=True)
        entry["events"][slot].record()
        entry["slot"] = entry["pending"] = slot

    def _comm_split(self, config, workspace):
        """(forward, backward) communication-SM counts for the next launch."""
        base = (config.fwd_num_comm_sms, config.bwd_num_comm_sms)
        if config.imbalance_comm_sms is None:
            return base
        capturing = torch.cuda.is_current_stream_capturing()
        entry = self._load_entry(workspace, create=not capturing)
        if entry is None:
            return base
        pending = entry["pending"]
        if pending is not None and not capturing and entry["events"][pending].query():
            entry["loads"] = entry["host"][pending].tolist()
            entry["pending"] = None
        split = select_comm_split(
            entry["loads"], base, config.imbalance_comm_sms, config.imbalance_threshold
        )
        if split != entry["split"]:
            entry["split"] = split
            if os.environ.get("CAKE_MOK_COMM_SPLIT_LOG", "0") not in ("", "0"):
                loads = entry["loads"]
                total = sum(loads)
                ratio = f"{max(loads) * len(loads) / total:.3f}" if total else "n/a"
                print(
                    f"[cake_mok_bf16 rank {workspace.ep_rank}] communication split -> "
                    f"{split} (max/mean routed load {ratio})",
                    file=sys.stderr,
                    flush=True,
                )
        return split

    @staticmethod
    def _schedule(schedule):
        return (
            schedule.peer_rank,
            schedule.peer_token_idx,
            schedule.num_tokens,
            schedule.tokens_per_expert,
        )

    def _weights(self, x, shared, routed):
        """BF16 shared weights; routed weights BF16 or native MXFP8 tuples.

        Returns True for MXFP8 routed experts. Routed MXFP8 tuples are
        validated by the kernels (``(data, scales)`` for forward/recompute,
        ``(data, scales, data_t, scales_t)`` gate/up and ``(data_t, scales_t)``
        down for backward).
        """
        kinds = {isinstance(weight, tuple) for weight in routed}
        if len(kinds) != 1:
            raise TypeError(
                "Routed weights must all be BF16 tensors or all MXFP8 tuples"
            )
        mxfp8 = kinds.pop()
        flat = [
            w
            for weight in (*shared, *routed)
            for w in (weight if isinstance(weight, tuple) else (weight,))
        ]
        if any(
            not isinstance(w, torch.Tensor)
            or w.device != x.device
            or not w.is_contiguous()
            for w in flat
        ):
            raise ValueError(
                "Expert weights must be contiguous CUDA tensors on the input device"
            )
        if any(w.dtype != torch.bfloat16 for w in shared) or (
            not mxfp8 and any(w.dtype != torch.bfloat16 for w in routed)
        ):
            raise ValueError("Shared and BF16 routed expert weights must be BF16")
        if shared[0].ndim != 2:
            raise ValueError("Shared gate weights must be a matrix")
        intermediate, hidden = shared[0].shape
        shapes = (
            (intermediate, hidden),
            (intermediate, hidden),
            (hidden, intermediate),
        )[: len(shared)]
        if (
            hidden != x.shape[1]
            or intermediate <= 0
            or intermediate % 256
            or any(tuple(w.shape) != s for w, s in zip(shared, shapes, strict=True))
        ):
            raise ValueError(
                "Expert weight shapes must match the prepared workspace and 256-column tiles"
            )
        if not mxfp8:
            routed_shapes = (
                (self.local_experts, intermediate, hidden),
                (self.local_experts, intermediate, hidden),
                (self.local_experts, hidden, intermediate),
            )[: len(routed)]
            if any(
                tuple(w.shape) != s for w, s in zip(routed, routed_shapes, strict=True)
            ):
                raise ValueError(
                    "Routed expert weight shapes must match the prepared local experts"
                )
        elif routed[0][0].ndim != 3 or routed[0][0].shape[0] != self.local_experts:
            raise ValueError(
                "MXFP8 routed weights must hold the prepared local experts"
            )
        return mxfp8

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
        mxfp8 = self._weights(
            x,
            (shared_gate_weights, shared_up_weights, shared_down_weights),
            (routed_gate_weights, routed_up_weights, routed_down_weights),
        )
        shared_rows = self._shared_rows(source_count)
        self._copy_source(workspace.x_buffer, x, 0, shared_rows)
        self._copy_source(
            workspace.router_weight_buffer, router_weights, 1, shared_rows
        )
        workspace.combine_buffer[
            source_count * workspace.topk : shared_rows * workspace.topk
        ].zero_()
        self._barrier(workspace)
        forward_kernel, _ = self._kernels(swiglu_limit, mxfp8)
        fwd_comm_sms, _ = self._comm_split(config, workspace)
        values = forward_kernel(
            workspace.x_buffer[:shared_rows],
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
            fwd_comm_sms,
            config.macrobatch_size,
            config.minibatch_size,
        )
        self._barrier(workspace)
        output = self.epilogues.forward(
            values[7],
            workspace.combine_buffer[: shared_rows * workspace.topk],
            workspace.router_weight_buffer[:shared_rows],
        )
        context = MoKForwardContext(
            x_routed=values[0],
            gate_shared=values[1],
            gate_routed=values[2],
            up_shared=values[3],
            up_routed=values[4],
            hidden_shared=values[5],
            hidden_routed=values[6],
            schedule=schedule,
        )
        return output[:source_count], context

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
        """Rebuild the backward context for activation checkpointing.

        Native ``recompute_forward_context`` semantics: dispatch plus shared and
        routed gate/up GEMMs and SwiGLU for the resident first macrobatch; no
        down projection, combine or output epilogue. Pass the same schedule,
        inputs, weights and ``swiglu_limit`` as the checkpointed forward; the
        returned context is interchangeable with that forward's context.
        """
        workspace, source_count = self._inputs(config, workspace, schedule, x, None)
        self._check_device(workspace)
        mxfp8 = self._weights(
            x,
            (shared_gate_weights, shared_up_weights),
            (routed_gate_weights, routed_up_weights),
        )
        shared_rows = self._shared_rows(source_count)
        self._copy_source(workspace.x_buffer, x, 0, shared_rows)
        self._barrier(workspace)
        fwd_comm_sms, _ = self._comm_split(config, workspace)
        values = self._recompute_kernel(swiglu_limit, mxfp8)(
            workspace.x_buffer[:shared_rows],
            workspace.x_buffer_ptrs,
            shared_gate_weights,
            routed_gate_weights,
            shared_up_weights,
            routed_up_weights,
            *self._schedule(schedule),
            workspace.topk,
            swiglu_limit,
            fwd_comm_sms,
            config.macrobatch_size,
            config.minibatch_size,
        )
        # Peers read this rank's x_buffer during dispatch.
        self._barrier(workspace)
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
        weight_grad_accumulators=None,
    ):
        """``weight_grad_accumulators``: with ``fp32_wgrad=True``, six FP32
        tensors (shared gate, up, down, routed gate, up, down) to which this
        call adds the weight gradients in FP32; the same tensors are returned.
        Routed macrobatch contributions are added in a fixed order."""
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
        mxfp8 = self._weights(
            x,
            (shared_gate_weights, shared_up_weights, shared_down_weights),
            (routed_gate_weights, routed_up_weights, routed_down_weights),
        )
        if mxfp8 != isinstance(forward_context.x_routed, tuple):
            raise ValueError("Forward context and routed weights differ in precision")
        shared_rows = self._shared_rows(source_count)
        self._copy_source(workspace.d_y_buffer, grad_output, 0, shared_rows)
        # Peers read this rank's x_buffer only to replay a second macrobatch.
        # Without one, the shared-expert path reads the caller's x when it
        # needs no padding rows.
        if (
            workspace.schedule_capacity <= config.macrobatch_size
            and shared_rows == source_count
        ):
            x_local = x
        else:
            self._copy_source(workspace.x_buffer, x, 0, shared_rows)
            x_local = workspace.x_buffer[:shared_rows]
        self._copy_source(
            workspace.router_weight_buffer, router_weights, 1, shared_rows
        )
        workspace.d_x_routed_buffer[
            source_count * workspace.topk : shared_rows * workspace.topk
        ].zero_()
        workspace.d_router_weight_buffer[source_count:shared_rows].zero_()
        self._barrier(workspace)
        _, backward_kernel = self._kernels(swiglu_limit, mxfp8)
        _, bwd_comm_sms = self._comm_split(config, workspace)
        values = backward_kernel(
            workspace.d_y_buffer[:shared_rows],
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
            x_local,
            workspace.x_buffer_ptrs,
            *self._schedule(schedule),
            workspace.topk,
            swiglu_limit,
            bwd_comm_sms,
            config.macrobatch_size,
            config.minibatch_size,
            weight_grad_accumulators=weight_grad_accumulators,
        )
        self._barrier(workspace)
        dx = self.epilogues.backward(
            values[0], workspace.d_x_routed_buffer[: shared_rows * workspace.topk]
        )
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
