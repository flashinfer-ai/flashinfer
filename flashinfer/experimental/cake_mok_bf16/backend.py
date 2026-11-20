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

from dataclasses import dataclass
import math
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
    _ffi_stream,
    _recall,
    _remember,
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


def _check_input(name, tensor, dtype, shape, device_index):
    if (
        not tensor.is_cuda
        or tensor.get_device() != device_index
        or tensor.dtype != dtype
        or not tensor.is_contiguous()
    ):
        raise ValueError(
            f"{name} must be contiguous {dtype} on the workspace CUDA device"
        )
    if tensor.shape != shape:
        raise ValueError(f"{name} shape does not match the workspace")


def _validate_inputs(config, workspace, schedule, x, router_weights, grad_output, rows):
    """``workspace.validate_inputs`` with the source row count ``rows`` given
    explicitly (the variable-source wrapper's logical count), so no logical
    workspace copy is built per call. Same checks, order and messages."""
    if not isinstance(config, native.MoKConfig):
        raise TypeError("config must be a MoKConfig")
    if not isinstance(workspace, native.MoKWorkspace):
        raise TypeError("workspace must be a MoKWorkspace")
    if not isinstance(schedule, native.MoKSchedule):
        raise TypeError("schedule must be a MoKSchedule")
    index = workspace.device.index
    activation = (rows, workspace.hidden_size)
    _check_input("x", x, torch.bfloat16, activation, index)
    if grad_output is not None:
        _check_input("grad_output", grad_output, torch.bfloat16, activation, index)
    if router_weights is not None:
        _check_input(
            "router_weights",
            router_weights,
            torch.float32,
            (rows, workspace.topk),
            index,
        )
    if schedule.peer_rank.numel() != workspace.schedule_capacity:
        raise ValueError("schedule capacity does not match the workspace")


def _weight_key(weight):
    """Validation-memo identity of one weight: storage pointer, shape, strides,
    dtype and device (MXFP8 tuples element-wise)."""
    if isinstance(weight, tuple):
        return tuple(map(_weight_key, weight))
    return (
        weight.data_ptr(),
        weight.shape,
        weight.stride(),
        weight.dtype,
        weight.get_device(),
    )


@dataclass(slots=True, eq=False)
class _StepPlan:
    """Launch plan of one adapter operation on one workspace (see
    ``MoKFunctional._plan``): fixed views of the workspace buffers the
    operation stages into, the fused-kernel, barrier and epilogue launches."""

    fused: object
    barrier: object
    epilogue: object
    x_rows: torch.Tensor
    scores: torch.Tensor | None = None
    dy_rows: torch.Tensor | None = None


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

    Reference form. The production path reads the owner bins the count kernel
    writes into the tail of its ``counts`` buffer (``MoKScheduler.owner_loads``);
    the two agree exactly (``tests/experimental/test_mok_bf16.py``).
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


def select_multimacro(tokens, macro_size):
    """Whether a rank launches the sm_107a multimacro MXFP8 forward program.

    ``tokens`` is the rank's observed schedule row count (the padded
    ``num_tokens`` the forward kernel reads), ``None`` before any observation.
    True exactly when that schedule holds two macro batches, the case the
    program speeds up; the two programs give bitwise the same results for any
    schedule, so a stale observation costs time only. Host-only.
    """
    return tokens is not None and -(-int(tokens) // macro_size) == 2


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
        self._context_free_forward_kernels = {}
        # id(native workspace) -> asynchronous per-rank routed-load observation.
        self._load_state = {}
        from .jit import target_arch

        self._multimacro_target = target_arch(self.device) == "sm_107a"
        self._sm_count = {}  # device index -> multi_processor_count
        # Validated weight sets (LRU, ``_weight_key`` per weight): one entry
        # per MoE layer and operation, no reference to any weight.
        self._validated = {}
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

    def _context_free_forward_kernel(self, swiglu_limit, mxfp8=False):
        """The context-free forward of a SwiGLU variant (BF16 routed experts)."""
        if mxfp8:
            raise ValueError(
                "save_context=False requires BF16 routed experts; the MXFP8 "
                "forward always saves its context"
            )
        key = swiglu_limit is not None
        if key not in self._context_free_forward_kernels:
            if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Call prepare(..., save_context=False) before CUDA Graph capture"
                )
            self._context_free_forward_kernels[key] = MoKForward(key, context=False)
        return self._context_free_forward_kernels[key]

    def _recompute_kernel(self, swiglu_limit, mxfp8=False):
        key = (mxfp8, swiglu_limit is not None)
        if key not in self._recompute_kernels:
            if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Call prepare_recompute() before CUDA Graph capture")
            recompute = MoKRecomputeMXFP8 if mxfp8 else MoKRecompute
            self._recompute_kernels[key] = recompute(key[1])
        return self._recompute_kernels[key]

    def prepare(self, swiglu_limit=None, mxfp8=False, *, save_context=True):
        """Compile the forward/backward kernels of this variant now.

        ``save_context=False`` also compiles the context-free forward that
        ``forward(..., save_context=False)`` runs (BF16 routed experts only).
        """
        if type(save_context) is not bool:
            raise TypeError("save_context must be a boolean")
        self._kernels(swiglu_limit, mxfp8)
        if not save_context:
            self._context_free_forward_kernel(swiglu_limit, mxfp8)

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
            # The native checks against the logical source count; no buffer is
            # allocated or resized and the shared workspace is not mutated.
            _validate_inputs(
                config, storage, schedule, x, router_weights, grad_output, count
            )
            return storage, count
        if isinstance(schedule, MoKSourceSchedule):
            raise ValueError(
                "A variable-source schedule requires its owning workspace wrapper"
            )
        if not isinstance(workspace, native.MoKWorkspace):
            native.validate_inputs(
                config, workspace, schedule, x, router_weights, grad_output
            )
        count = workspace.num_local_tokens
        _validate_inputs(
            config, workspace, schedule, x, router_weights, grad_output, count
        )
        return workspace, count

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
        sms = self._sm_count.get(workspace.device.index)
        if sms is None:
            sms = self._sm_count[workspace.device.index] = (
                torch.cuda.get_device_properties(workspace.device).multi_processor_count
            )
        for count in (config.fwd_num_comm_sms, config.bwd_num_comm_sms):
            if type(count) is not int or count <= 0 or count % 2 or count >= sms:
                raise ValueError(
                    "Communication-SM counts must be positive, even and leave compute SMs"
                )
        if config.imbalance_comm_sms is not None:
            hot = config.imbalance_comm_sms
            if (
                type(hot) is not tuple
                or len(hot) != 2
                or any(
                    type(count) is not int or count <= 0 or count % 2 or count >= sms
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
        if type(config.comm_split_log) is not bool:
            raise TypeError("comm_split_log must be a boolean")
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
        self._observe_load(workspace, config, count)
        layout = self.scheduler.layout(counts, workspace.schedule_capacity, rank)
        self.scheduler.release_counts()
        if owner is not None:
            return MoKSourceSchedule(
                peer_rank=rank,
                peer_token_idx=token,
                num_tokens=count,
                tokens_per_expert=counts,
                layout=layout,
                num_source_tokens=top_experts.shape[0],
                workspace=owner,
            )
        return native.MoKSchedule(
            peer_rank=rank,
            peer_token_idx=token,
            num_tokens=count,
            tokens_per_expert=counts,
            layout=layout,
        )

    def _load_entry(self, workspace, create):
        entry = self._load_state.get(id(workspace))
        if entry is None and create:
            entry = {
                "host": [
                    torch.empty(workspace.ep_size, dtype=torch.int32, pin_memory=True)
                    for _ in range(2)
                ],
                "events": [torch.cuda.Event() for _ in range(2)],
                "slot": 0,
                "pending": None,
                "loads": (),
                "split": None,
                "tokens_host": [
                    torch.empty(1, dtype=torch.int32, pin_memory=True) for _ in range(2)
                ],
                "tokens_observed": [False, False],
                "tokens": None,
                "multimacro": False,
            }
            self._load_state[id(workspace)] = entry
        return entry

    def _observe_load(self, workspace, config, num_tokens=None):
        """Queue an asynchronous observation of the per-rank routed-row load.

        Every rank counts the gathered expert IDs per owning rank on device
        (the scheduler's count kernel bins them in the pass that counts this
        rank's experts; padding and invalid IDs excluded, as
        ``count_routed_load`` specifies) and copies the counts into pinned
        host memory behind a CUDA event: no host synchronization.
        ``_comm_split`` reads the latest completed observation, so a split
        change takes effect from the first later launch whose event has
        completed; a completed observation is also adopted here, before the
        next one replaces it. The gathered IDs, and so the counts, are
        identical on every rank, but each rank decides from its own latest
        completed observation: with back-to-back steps a launch may run with different
        splits on different ranks for a step. The split is a per-rank launch
        parameter (how many of that rank's SMs run its communication program);
        the protocol does not require the ranks to agree on it. Skipped during
        CUDA Graph capture: captured launches keep the split in force at
        capture time.

        Behind the same event it copies this rank's schedule row count
        ``num_tokens`` (the scheduler's ``(1,)`` int32) for the forward program
        choice (``_multimacro``) whenever that choice is open (sm_107a,
        schedule capacity above one macro batch), with or without
        ``imbalance_comm_sms``.
        """
        rows = num_tokens is not None and self._multimacro_open(config, workspace)
        if (
            config.imbalance_comm_sms is None and not rows
        ) or torch.cuda.is_current_stream_capturing():
            return
        entry = self._load_entry(workspace, create=True)
        # Keep a completed observation before its slot stops being pending:
        # when the host runs ahead of the device, no launch of the previous
        # step may have found it complete.
        self._collect_load(entry)
        loads = self.scheduler.owner_loads()
        slot = entry["slot"] ^ 1
        entry["host"][slot].copy_(loads, non_blocking=True)
        if rows:
            entry["tokens_host"][slot].copy_(num_tokens, non_blocking=True)
        entry["tokens_observed"][slot] = rows
        entry["events"][slot].record()
        entry["slot"] = entry["pending"] = slot

    @staticmethod
    def _collect_load(entry):
        """Adopt the pending observation once its event has completed."""
        pending = entry["pending"]
        if pending is not None and entry["events"][pending].query():
            entry["loads"] = entry["host"][pending].tolist()
            if entry["tokens_observed"][pending]:
                entry["tokens"] = int(entry["tokens_host"][pending][0])
            entry["pending"] = None

    def _multimacro_open(self, config, workspace):
        """Static guard of the forward program choice: sm_107a and a schedule
        capacity above one macro batch (otherwise every schedule has at most
        one macro batch and the default program always runs)."""
        return (
            self._multimacro_target
            and -(-workspace.schedule_capacity // config.macrobatch_size) > 1
        )

    def _multimacro(self, config, workspace):
        """Whether this rank's next MXFP8 forward launches the multimacro program.

        Decided per rank from its latest completed observation
        (``select_multimacro``), like the communication split: ranks may run
        different programs for a step, which the protocol permits (the
        programs differ only in the order of this rank's own work and give
        bitwise the same results). No observation yet keeps the default
        program; during CUDA Graph capture the choice in force is kept.
        ``config.comm_split_log`` also prints each change of the choice.
        """
        if not self._multimacro_open(config, workspace):
            return False
        capturing = torch.cuda.is_current_stream_capturing()
        entry = self._load_entry(workspace, create=not capturing)
        if entry is None:
            return False
        if not capturing:
            self._collect_load(entry)
        choice = select_multimacro(entry["tokens"], config.macrobatch_size)
        if choice != entry["multimacro"]:
            entry["multimacro"] = choice
            if config.comm_split_log:
                print(
                    f"[cake_mok_bf16 rank {workspace.ep_rank}] forward program -> "
                    f"{'multimacro' if choice else 'default'} (schedule rows "
                    f"{entry['tokens']}, macro batch {config.macrobatch_size})",
                    file=sys.stderr,
                    flush=True,
                )
        return choice

    def _comm_split(self, config, workspace):
        """(forward, backward) communication-SM counts for the next launch.

        ``config.comm_split_log`` prints each change of the split to stderr.
        """
        base = (config.fwd_num_comm_sms, config.bwd_num_comm_sms)
        if config.imbalance_comm_sms is None:
            return base
        capturing = torch.cuda.is_current_stream_capturing()
        entry = self._load_entry(workspace, create=not capturing)
        if entry is None:
            return base
        if not capturing:
            self._collect_load(entry)
        split = select_comm_split(
            entry["loads"], base, config.imbalance_comm_sms, config.imbalance_threshold
        )
        if split != entry["split"]:
            entry["split"] = split
            if config.comm_split_log:
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
        """:meth:`_validate_weights`, memoised per weight set.

        The key is the device and width of ``x`` plus every weight's storage
        pointer, shape, strides, dtype and device, so a hit is a weight set
        that validated before under exactly these properties. The memo holds
        no tensor (a freed weight costs nothing) and is an LRU of
        ``_GEOMETRY_LIMIT`` entries (MoE layers x operations). Unsupported
        argument types skip the memo and raise from the full validation.
        """
        try:
            key = (
                x.get_device(),
                x.shape[1],
                *map(_weight_key, shared),
                *map(_weight_key, routed),
            )
        except (AttributeError, TypeError):
            key = None
        if key is not None:
            hit = _recall(self._validated, key)
            if hit is not None:
                return hit[0]
        mxfp8 = self._validate_weights(x, shared, routed)
        if key is not None:
            _remember(self._validated, key, (mxfp8,))
        return mxfp8

    def _validate_weights(self, x, shared, routed):
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

    def _plan(self, workspace, key, build):
        """The workspace's launch plan for ``key``, built (validated) on first use.

        Plans live in ``workspace.launch_plans`` and so die with the
        workspace. A key names only per-layer-stable state: the operation and
        its prepared launcher (variant), the source rows, the communication
        split, the SwiGLU limit and the geometry. Weights, inputs, outputs and
        schedules are per-call arguments, never part of a plan; all MoE
        layers of one geometry share a plan. Captured graphs keep using the
        plan's counter slab, which the workspace's lifetime contract covers;
        plans are therefore never evicted (their number is bounded by the
        distinct geometries and splits, not by the number of layers).
        """
        plans = workspace.launch_plans
        plan = plans.get(key)
        if plan is None:
            plan = plans[key] = build()
        return plan

    def _barrier_plan(self, workspace):
        key = ("barrier", self.communication)
        plans = workspace.launch_plans
        if key not in plans:
            plans[key] = self.communication.barrier_plan(
                workspace.barrier_buffer,
                workspace.barrier_buffer_ptrs,
                workspace.barrier_buffer_multicast_ptr,
                workspace.barrier_target,
            )
        return plans[key]

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
        *,
        save_context=True,
    ):
        """Returns ``(y, context)``.

        ``save_context=False`` (BF16 routed experts) runs the context-free
        forward of an activation-checkpointed step: it stores no gate/up
        context, returns ``(y, None)`` with ``y`` bitwise identical to the
        context forward's, and the step rebuilds the context with
        ``recompute_forward_context`` (same schedule, inputs, weights and
        ``swiglu_limit``) before ``backward``. Compile it with
        ``prepare(swiglu_limit, save_context=False)`` before CUDA Graph capture.
        """
        if type(save_context) is not bool:
            raise TypeError("save_context must be a boolean")
        workspace, source_count = self._inputs(
            config, workspace, schedule, x, router_weights
        )
        self._check_device(workspace)
        weights = (
            shared_gate_weights,
            shared_up_weights,
            shared_down_weights,
            routed_gate_weights,
            routed_up_weights,
            routed_down_weights,
        )
        mxfp8 = self._weights(x, weights[:3], weights[3:])
        if mxfp8 and not save_context:
            raise ValueError(
                "save_context=False requires BF16 routed experts; the MXFP8 "
                "forward always saves its context"
            )
        shared_rows = self._shared_rows(source_count)
        if mxfp8 or schedule.layout is None:
            return self._forward_native(
                config, workspace, schedule, x, router_weights, weights,
                swiglu_limit, source_count, shared_rows, mxfp8, save_context,
            )  # fmt: skip
        topk = workspace.topk
        if save_context:
            forward_kernel, _ = self._kernels(swiglu_limit, False)
        else:
            # A launcher of its own, so its plans are keyed apart.
            forward_kernel = self._context_free_forward_kernel(swiglu_limit)
        fwd_comm_sms, _ = self._comm_split(config, workspace)
        intermediate = shared_gate_weights.shape[0]
        experts = schedule.tokens_per_expert.numel()

        def build():
            x_rows = workspace.x_buffer[:shared_rows]
            scores = workspace.router_weight_buffer[:shared_rows]
            return _StepPlan(
                fused=forward_kernel.plan(
                    x_rows,
                    workspace.x_buffer_ptrs,
                    workspace.combine_buffer_ptrs,
                    intermediate,
                    experts,
                    workspace.schedule_capacity,
                    topk,
                    swiglu_limit,
                    fwd_comm_sms,
                    config.macrobatch_size,
                    config.minibatch_size,
                ),
                barrier=self._barrier_plan(workspace),
                epilogue=self.epilogues.plan(
                    workspace.combine_buffer[: shared_rows * topk], scores
                ),
                x_rows=x_rows,
                scores=scores,
            )

        plan = self._plan(
            workspace,
            (
                "forward",
                forward_kernel,
                shared_rows,
                fwd_comm_sms,
                swiglu_limit,
                config.macrobatch_size,
                config.minibatch_size,
                intermediate,
                experts,
            ),
            build,
        )
        barrier = plan.barrier
        with _ffi_stream():
            # Stage: counters, source rows, then the peer barrier (B2).
            plan.fused.slab.zero_()
            if shared_rows > source_count:
                self._copy_source(workspace.x_buffer, x, 0, shared_rows)
                self._copy_source(
                    workspace.router_weight_buffer, router_weights, 1, shared_rows
                )
                workspace.combine_buffer[
                    source_count * topk : shared_rows * topk
                ].zero_()
                scores = plan.scores
            else:
                plan.x_rows.copy_(x)
                # Peers never read the router weights in forward and no row
                # needs padding: the epilogue reads the caller's tensor.
                scores = router_weights
            if barrier is not None:
                barrier.launch(())
            values = forward_kernel.launch_planned(
                plan.fused,
                shared_gate_weights,
                routed_gate_weights,
                shared_up_weights,
                routed_up_weights,
                shared_down_weights,
                routed_down_weights,
                schedule.peer_rank,
                schedule.peer_token_idx,
                schedule.num_tokens,
                schedule.layout,
            )
            if barrier is not None:
                barrier.launch(())
            output = self.epilogues.launch_planned(plan.epilogue, values[7], scores)
        if not save_context:
            return output[:source_count], None
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

    def _forward_native(
        self,
        config,
        workspace,
        schedule,
        x,
        router_weights,
        weights,
        swiglu_limit,
        source_count,
        shared_rows,
        mxfp8,
        save_context,
    ):
        """Unplanned forward (MXFP8 routed experts, or a schedule without
        layout): the launchers' own validation and launch path."""
        (
            shared_gate_weights,
            shared_up_weights,
            shared_down_weights,
            routed_gate_weights,
            routed_up_weights,
            routed_down_weights,
        ) = weights
        self._copy_source(workspace.x_buffer, x, 0, shared_rows)
        self._copy_source(
            workspace.router_weight_buffer, router_weights, 1, shared_rows
        )
        if shared_rows > source_count:
            workspace.combine_buffer[
                source_count * workspace.topk : shared_rows * workspace.topk
            ].zero_()
        self._barrier(workspace)
        if save_context:
            forward_kernel, _ = self._kernels(swiglu_limit, mxfp8)
        else:
            forward_kernel = self._context_free_forward_kernel(swiglu_limit)
        fwd_comm_sms, _ = self._comm_split(config, workspace)
        program = {"multimacro": self._multimacro(config, workspace)} if mxfp8 else {}
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
            layout=schedule.layout,
            **program,
        )
        self._barrier(workspace)
        output = self.epilogues.forward(
            values[7],
            workspace.combine_buffer[: shared_rows * workspace.topk],
            workspace.router_weight_buffer[:shared_rows],
        )
        if not save_context:
            return output[:source_count], None
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
        returned context is interchangeable with that forward's context. It is
        the only context source after ``forward(..., save_context=False)``.
        """
        workspace, source_count = self._inputs(config, workspace, schedule, x, None)
        self._check_device(workspace)
        weights = (
            shared_gate_weights,
            shared_up_weights,
            routed_gate_weights,
            routed_up_weights,
        )
        mxfp8 = self._weights(x, weights[:2], weights[2:])
        shared_rows = self._shared_rows(source_count)
        if mxfp8 or schedule.layout is None:
            return self._recompute_native(
                config, workspace, schedule, x, weights, swiglu_limit,
                shared_rows, mxfp8,
            )  # fmt: skip
        kernel = self._recompute_kernel(swiglu_limit, False)
        fwd_comm_sms, _ = self._comm_split(config, workspace)
        intermediate = shared_gate_weights.shape[0]
        experts = schedule.tokens_per_expert.numel()

        def build():
            x_rows = workspace.x_buffer[:shared_rows]
            return _StepPlan(
                fused=kernel.plan(
                    x_rows,
                    workspace.x_buffer_ptrs,
                    intermediate,
                    experts,
                    workspace.schedule_capacity,
                    workspace.topk,
                    swiglu_limit,
                    fwd_comm_sms,
                    config.macrobatch_size,
                    config.minibatch_size,
                ),
                barrier=self._barrier_plan(workspace),
                epilogue=None,
                x_rows=x_rows,
            )

        plan = self._plan(
            workspace,
            (
                "recompute",
                kernel,
                shared_rows,
                fwd_comm_sms,
                swiglu_limit,
                config.macrobatch_size,
                config.minibatch_size,
                intermediate,
                experts,
            ),
            build,
        )
        barrier = plan.barrier
        with _ffi_stream():
            plan.fused.slab.zero_()
            if shared_rows > source_count:
                self._copy_source(workspace.x_buffer, x, 0, shared_rows)
            else:
                plan.x_rows.copy_(x)
            if barrier is not None:
                barrier.launch(())
            values = kernel.launch_planned(
                plan.fused,
                shared_gate_weights,
                routed_gate_weights,
                shared_up_weights,
                routed_up_weights,
                schedule.peer_rank,
                schedule.peer_token_idx,
                schedule.num_tokens,
                schedule.layout,
            )
            # Peers read this rank's x_buffer during dispatch.
            if barrier is not None:
                barrier.launch(())
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

    def _recompute_native(
        self, config, workspace, schedule, x, weights, swiglu_limit, shared_rows, mxfp8
    ):
        """Unplanned recompute (see :meth:`_forward_native`)."""
        (
            shared_gate_weights,
            shared_up_weights,
            routed_gate_weights,
            routed_up_weights,
        ) = weights
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
            layout=schedule.layout,
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
        weights = (
            shared_gate_weights,
            shared_up_weights,
            shared_down_weights,
            routed_gate_weights,
            routed_up_weights,
            routed_down_weights,
        )
        mxfp8 = self._weights(x, weights[:3], weights[3:])
        if mxfp8 != isinstance(forward_context.x_routed, tuple):
            raise ValueError("Forward context and routed weights differ in precision")
        shared_rows = self._shared_rows(source_count)
        if mxfp8 or schedule.layout is None:
            return self._backward_native(
                config, workspace, schedule, forward_context, grad_output, x,
                router_weights, weights, swiglu_limit, weight_grad_accumulators,
                source_count, shared_rows, mxfp8,
            )  # fmt: skip
        topk = workspace.topk
        _, backward_kernel = self._kernels(swiglu_limit, False)
        _, bwd_comm_sms = self._comm_split(config, workspace)
        backward_kernel.check_accumulators(weights, weight_grad_accumulators)
        intermediate = shared_gate_weights.shape[0]
        experts = schedule.tokens_per_expert.numel()

        def build():
            dy_rows = workspace.d_y_buffer[:shared_rows]
            return _StepPlan(
                fused=backward_kernel.plan(
                    dy_rows,
                    workspace.d_y_buffer_ptrs,
                    workspace.d_x_routed_buffer_ptrs,
                    workspace.router_weight_buffer_ptrs,
                    workspace.d_router_weight_buffer_ptrs,
                    workspace.x_buffer_ptrs,
                    intermediate,
                    experts,
                    workspace.schedule_capacity,
                    topk,
                    swiglu_limit,
                    bwd_comm_sms,
                    config.macrobatch_size,
                    config.minibatch_size,
                ),
                barrier=self._barrier_plan(workspace),
                epilogue=self.epilogues.plan(
                    workspace.d_x_routed_buffer[: shared_rows * topk], False
                ),
                x_rows=workspace.x_buffer[:shared_rows],
                scores=workspace.router_weight_buffer[:shared_rows],
                dy_rows=dy_rows,
            )

        plan = self._plan(
            workspace,
            (
                "backward",
                backward_kernel,
                shared_rows,
                bwd_comm_sms,
                swiglu_limit,
                config.macrobatch_size,
                config.minibatch_size,
                intermediate,
                experts,
            ),
            build,
        )
        barrier = plan.barrier
        padded = shared_rows > source_count
        context = forward_context
        with _ffi_stream():
            plan.fused.slab.zero_()
            if padded:
                self._copy_source(workspace.d_y_buffer, grad_output, 0, shared_rows)
            else:
                plan.dy_rows.copy_(grad_output)
            # Peers read this rank's x_buffer only to replay a second macrobatch.
            # Without one, the shared-expert path reads the caller's x when it
            # needs no padding rows.
            if workspace.schedule_capacity <= config.macrobatch_size and not padded:
                x_local = x
            else:
                if padded:
                    self._copy_source(workspace.x_buffer, x, 0, shared_rows)
                else:
                    plan.x_rows.copy_(x)
                x_local = plan.x_rows
            if padded:
                self._copy_source(
                    workspace.router_weight_buffer, router_weights, 1, shared_rows
                )
                workspace.d_x_routed_buffer[
                    source_count * topk : shared_rows * topk
                ].zero_()
                workspace.d_router_weight_buffer[source_count:shared_rows].zero_()
            else:
                plan.scores.copy_(router_weights)
            if barrier is not None:
                barrier.launch(())
            values = backward_kernel.launch_planned(
                plan.fused,
                shared_gate_weights,
                routed_gate_weights,
                shared_up_weights,
                routed_up_weights,
                shared_down_weights,
                routed_down_weights,
                context.x_routed,
                context.gate_shared,
                context.gate_routed,
                context.up_shared,
                context.up_routed,
                context.hidden_shared,
                context.hidden_routed,
                x_local,
                schedule.peer_rank,
                schedule.peer_token_idx,
                schedule.num_tokens,
                schedule.tokens_per_expert,
                schedule.layout,
                weight_grad_accumulators,
            )
            # Only dx_shared and the six weight gradients outlive the launch;
            # the scratch is dropped before the epilogue output and the
            # score-gradient copy are allocated (see ``_backward_native``).
            dx_shared, weight_grads = values[0], values[9:]
            del values
            if barrier is not None:
                barrier.launch(())
            dx = self.epilogues.launch_planned(plan.epilogue, dx_shared)
            del dx_shared
        dscores = workspace.d_router_weight_buffer[:source_count].clone()
        dwg_shared, dwg_routed, dwu_shared, dwu_routed, dwd_shared, dwd_routed = (
            weight_grads
        )
        return (
            dx[:source_count],
            dscores,
            dwg_routed,
            dwu_routed,
            dwd_routed,
            dwg_shared,
            dwu_shared,
            dwd_shared,
        )

    def _backward_native(
        self,
        config,
        workspace,
        schedule,
        forward_context,
        grad_output,
        x,
        router_weights,
        weights,
        swiglu_limit,
        weight_grad_accumulators,
        source_count,
        shared_rows,
        mxfp8,
    ):
        """Unplanned backward (see :meth:`_forward_native`)."""
        (
            shared_gate_weights,
            shared_up_weights,
            shared_down_weights,
            routed_gate_weights,
            routed_up_weights,
            routed_down_weights,
        ) = weights
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
        if shared_rows > source_count:
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
            layout=schedule.layout,
        )
        # Only dx_shared and the six weight gradients outlive the launch. The
        # kernel's routed and shared scratch (values[1:9]) is dropped here,
        # before the epilogue output and the score-gradient copy are allocated,
        # instead of at return: the caching allocator hands its blocks only to
        # later allocations on this stream, which are ordered after the kernel.
        dx_shared, weight_grads = values[0], values[9:]
        del values
        self._barrier(workspace)
        dx = self.epilogues.backward(
            dx_shared, workspace.d_x_routed_buffer[: shared_rows * workspace.topk]
        )
        del dx_shared
        dscores = workspace.d_router_weight_buffer[:source_count].clone()
        dwg_shared, dwg_routed, dwu_shared, dwu_routed, dwd_shared, dwd_routed = (
            weight_grads
        )
        return (
            dx[:source_count],
            dscores,
            dwg_routed,
            dwu_routed,
            dwd_routed,
            dwg_shared,
            dwu_shared,
            dwd_shared,
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
