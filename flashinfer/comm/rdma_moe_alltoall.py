# Copyright (c) 2026 by FlashInfer team.
# Portions adapted from DeepEP (MIT); see LICENSE.deepep.txt and NOTICE.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""Expert-major RDMA MoE all-to-all communication on SM120."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, NamedTuple, Optional
from numbers import Real
import struct

import torch
import torch.distributed as dist
import tvm_ffi

from ..api_logging import flashinfer_api
from ..trace.templates.rdma_moe_alltoall import (
    rdma_moe_dispatch_trace,
    rdma_moe_combine_trace,
)
from .rdma_moe_alltoall_policy import select_rdma_strategy
from ..jit.rdma_moe_alltoall import get_gin_module, get_quantization_module

_PRECISION = {"fp8": 0, "nvfp4": 1, "bf16": 2}
_HIDDEN = (4096, 6144, 7168)
_PLANNED_REDUCTION = "bf16_local_partial_planned"


def _call(function, *args):
    # TensorView launches use TVMFFIEnvGetStream, including under CUDA capture.
    with tvm_ffi.use_torch_stream():
        return function(*args)


def _error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _plain(value):
    if hasattr(value, "items"):
        return {str(key): _plain(item) for key, item in value.items()}
    if not isinstance(value, (str, bytes, int, float, bool, type(None))) and hasattr(
        value, "__iter__"
    ):
        return [_plain(item) for item in value]
    return value


class DispatchHandle(tuple):
    """Tuple-compatible dispatch metadata, valid only on its owning buffer.

    The five entries are src_info, ranges, capacity, hidden and num_experts.
    Metadata tensors remain
    alive with the handle. Reusing those tensors via ``dispatch(out=...)``
    overwrites the corresponding old handle, as with any explicit output reuse.
    """

    _owner: object
    precision: str
    _combine_plan: Optional[_CombinePlan]

    def __new__(cls, values, *, owner, precision, combine_plan=None):
        result = super().__new__(cls, values)
        result._owner = owner
        result.precision = precision
        result._combine_plan = combine_plan
        return result


class _CombinePlan:
    """Per-dispatch storage; no pointers into the buffer's alternating arena.

    The native dispatch writes every used entry on each execution, including
    graph replay. Callers must not mutate these private tensors. A retained
    handle keeps its plan alive even after later dispatches reuse arena slots.
    """

    __slots__ = ("tensors", "tokens", "topk", "owner")

    def __init__(self, buffer, tokens, topk):
        world = buffer.num_ranks
        capacity = buffer.num_max_dispatch_tokens_per_rank
        shapes = (
            (world, capacity, 8),
            (world, capacity, 8),
            (world, capacity),
            (world,),
            (tokens, topk),
            (tokens, topk),
            (world, capacity),
        )
        dtypes = (
            torch.int64,
            torch.float32,
            torch.int32,
            torch.int32,
            torch.int32,
            torch.float32,
            torch.int32,
        )
        self.tensors = tuple(
            torch.empty(shape, dtype=dtype, device=buffer.device)
            for shape, dtype in zip(shapes, dtypes, strict=True)
        )
        self.tokens = tokens
        self.topk = topk
        self.owner = buffer._owner


@dataclass(frozen=True)
class RdmaMoeAlltoAllConfig:
    """Fixed communication precision and launch policy for an all-to-all buffer.

    ``communication_mode="auto"`` chooses dispatch and combine independently
    from the measured device profile and fixed token capacity. Explicit mode
    or launch overrides require ``communication_mode="legacy"`` or
    ``"coalesced"``. ``num_sms`` and ``num_contexts`` apply to both operations;
    operation-specific values take precedence.

    ``combine_reduction="fp32"`` accumulates in FP32 and rounds once to BF16.
    The two BF16 local-partial modes have a separate rounding contract described
    in :meth:`RdmaMoeAlltoAll.combine`. The planned mode exchanges routes and
    weights in dispatch and retains their snapshot in the returned handle.
    All ranks must construct the same configuration.
    """

    communication_dtype: str = "nvfp4"
    communication_mode: str = "auto"
    traffic_class: Optional[int] = None
    num_sms: Optional[int] = None
    num_contexts: Optional[int] = None
    dispatch_mode: Optional[str] = None
    combine_mode: Optional[str] = None
    dispatch_num_sms: Optional[int] = None
    combine_num_sms: Optional[int] = None
    dispatch_num_contexts: Optional[int] = None
    combine_num_contexts: Optional[int] = None
    combine_reduction: str = "fp32"


class DispatchResult(NamedTuple):
    """Expert-major payload, scales, valid row counts and its combine handle.

    When requested, ``recv_hook`` completes the receive phase on the same CUDA
    stream as the send phase; invoke it exactly once before consuming outputs.
    BF16 communication returns an empty last dimension in ``scales``.
    """

    data: torch.Tensor
    scales: torch.Tensor
    counts: torch.Tensor
    handle: DispatchHandle
    recv_hook: Optional[Callable[[], None]]


class CombineResult(NamedTuple):
    """BF16 token output and an optional same-stream receive hook."""

    data: torch.Tensor
    recv_hook: Optional[Callable[[], None]]


class RdmaMoeAlltoAll:
    """Collective expert dispatch and weighted combine over GDAKI RDMA.

    Parameters
    ----------
    group : torch.distributed.ProcessGroup or None
        Initialized expert-parallel group. ``None`` uses WORLD. Every rank must
        construct and close the buffer collectively on its assigned CUDA device.
    num_max_dispatch_tokens_per_rank : int
        Fixed token capacity per rank. Dispatch output tensors have shape
        ``[num_experts / world_size, world_size * capacity, payload_width]``.
    hidden : int
        Logical hidden width: 4096, 6144 or 7168.
    num_experts : int
        Total expert count, divisible by the process-group size.
    config : RdmaMoeAlltoAllConfig or None
        Fixed precision, reduction and launch policy. Defaults to NVFP4 with
        automatic launch selection and FP32 combine accumulation.

    Notes
    -----
    Requires SM120 GPUs, RDMA-capable NICs, and the validated NCCL/GDAKI build
    documented in ``docs/development/rdma_moe_alltoall_dependencies.md``.
    All non-self traffic uses RDMA, including traffic within one host.
    This expert-major primitive does not register a rank-major ``moe_ep`` backend.

    Operations enqueue on the current CUDA stream. A buffer requires ordered
    execution; changing streams requires caller-provided synchronization.
    Receive hooks run exactly once on their send stream, before another call.
    Input and output tensors must remain alive until queued GPU work completes.

    CUDA graphs must contain an even number of dispatch/combine operations
    (normally complete dispatch/combine pairs), preserving alternating workspace
    slots on serial replay. Arbitrary eager/graph or cross-graph interleaving is
    unsupported. Retire all graph users before collective :meth:`close`.
    """

    def __init__(
        self,
        group,
        num_max_dispatch_tokens_per_rank,
        hidden,
        num_experts,
        *,
        config: Optional[RdmaMoeAlltoAllConfig] = None,
    ):
        if not dist.is_initialized():
            raise RuntimeError(
                "Initialize torch.distributed before creating an RDMA MoE all-to-all buffer"
            )
        invalid_config = config is not None and not isinstance(
            config, RdmaMoeAlltoAllConfig
        )
        settings = (
            config
            if isinstance(config, RdmaMoeAlltoAllConfig)
            else RdmaMoeAlltoAllConfig()
        )
        communication_dtype = settings.communication_dtype
        communication_mode = settings.communication_mode
        traffic_class = settings.traffic_class
        num_sms = settings.num_sms
        num_contexts = settings.num_contexts
        dispatch_mode = settings.dispatch_mode
        combine_mode = settings.combine_mode
        dispatch_num_sms = settings.dispatch_num_sms
        combine_num_sms = settings.combine_num_sms
        dispatch_num_contexts = settings.dispatch_num_contexts
        combine_num_contexts = settings.combine_num_contexts
        combine_reduction = settings.combine_reduction
        self.config = settings
        self.group = group
        self.rank = dist.get_rank(group)
        self.num_ranks = dist.get_world_size(group)
        self.num_max_dispatch_tokens_per_rank = num_max_dispatch_tokens_per_rank
        self.hidden = hidden
        self.num_experts = num_experts
        self.communication_dtype = communication_dtype
        self.combine_reduction = combine_reduction
        self._handle = None
        self._closed = False
        self._pending = None
        self._owner = object()
        self._slot = 0
        mode = communication_mode
        if mode is None:
            mode = "auto"
        self._mode = mode
        collective_config = dict(
            capacity=num_max_dispatch_tokens_per_rank,
            hidden=hidden,
            num_experts=num_experts,
            communication_dtype=communication_dtype,
            communication_mode=mode,
            traffic_class=traffic_class,
            num_sms=num_sms,
            num_contexts=num_contexts,
            dispatch_mode=dispatch_mode,
            combine_mode=combine_mode,
            dispatch_num_sms=dispatch_num_sms,
            combine_num_sms=combine_num_sms,
            dispatch_num_contexts=dispatch_num_contexts,
            combine_num_contexts=combine_num_contexts,
            combine_reduction=combine_reduction,
        )
        budgets = {
            key: collective_config[key]
            for key in (
                "num_sms",
                "num_contexts",
                "dispatch_num_sms",
                "combine_num_sms",
                "dispatch_num_contexts",
                "combine_num_contexts",
            )
        }
        error = None
        profile = None
        try:
            if invalid_config:
                raise TypeError("config must be RdmaMoeAlltoAllConfig or None")
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "RDMA MoE all-to-all construction must happen outside CUDA graph capture"
                )
            for name, value in (
                ("capacity", num_max_dispatch_tokens_per_rank),
                ("hidden", hidden),
                ("num_experts", num_experts),
            ):
                if type(value) is not int or value <= 0:
                    raise ValueError(f"{name} must be a positive integer")
            if hidden not in _HIDDEN:
                raise ValueError(f"hidden must be one of {_HIDDEN}")
            if num_experts % self.num_ranks or num_experts // self.num_ranks > 1024:
                raise ValueError(
                    "experts must divide EP size with at most 1024 local experts"
                )
            if num_max_dispatch_tokens_per_rank * self.num_ranks >= 2**31:
                raise ValueError(
                    "dispatch capacity exceeds the kernel int32 index range"
                )
            if communication_dtype not in _PRECISION:
                raise ValueError(
                    "communication_dtype must be 'bf16', 'fp8', or 'nvfp4'"
                )
            if combine_reduction not in (
                "fp32",
                "bf16_local_partial",
                _PLANNED_REDUCTION,
            ):
                raise ValueError(
                    "combine_reduction must be fp32, bf16_local_partial, "
                    "or bf16_local_partial_planned"
                )
            if combine_reduction != "fp32" and communication_dtype != "bf16":
                raise ValueError("local-partial reduction requires BF16 communication")
            for name, value in budgets.items():
                if value is not None and (type(value) is not int or value <= 0):
                    raise ValueError(f"{name} must be None or a positive integer")
            if traffic_class is not None and (
                type(traffic_class) is not int or not 0 <= traffic_class <= 255
            ):
                raise ValueError("traffic_class must be None or an integer in [0,255]")
            self.device = torch.device("cuda", torch.cuda.current_device())
            properties = torch.cuda.get_device_properties(self.device)
            profile = (
                properties.name,
                (properties.major, properties.minor),
                properties.multi_processor_count,
            )
            if profile[1] != (12, 0):
                raise ValueError("RDMA MoE all-to-all supports SM120 only")
            for name in ("num_sms", "dispatch_num_sms", "combine_num_sms"):
                if budgets[name] is not None and budgets[name] > profile[2]:
                    raise ValueError(f"{name} exceeds the device SM count")
            for name in (
                "num_contexts",
                "dispatch_num_contexts",
                "combine_num_contexts",
            ):
                if (
                    budgets[name] is not None
                    and budgets[name] > num_experts // self.num_ranks
                ):
                    raise ValueError(f"{name} exceeds the local expert count")
            if mode not in ("legacy", "coalesced", "auto"):
                raise ValueError("GIN mode must be legacy, coalesced, or auto")
            if num_max_dispatch_tokens_per_rank * self.num_ranks % 4:
                raise ValueError("GIN requires world_size * capacity divisible by four")
            if any(
                value not in (None, "legacy", "coalesced")
                for value in (dispatch_mode, combine_mode)
            ):
                raise ValueError("GIN per-operation modes must be legacy or coalesced")
            if mode == "auto" and (
                dispatch_mode is not None
                or combine_mode is not None
                or any(value is not None for value in budgets.values())
            ):
                raise ValueError(
                    "auto cannot be combined with explicit mode or launch overrides"
                )
        except Exception as exc:
            error = _error(exc)
        records = self._gather((collective_config, profile, error))
        self._raise_errors(
            [record[2] for record in records], "validating configuration"
        )
        if any(record[0] != collective_config for record in records):
            raise ValueError(
                f"RDMA MoE all-to-all configuration differs across ranks: {[record[0] for record in records]}"
            )
        self._strategy_selection = dict(
            requested_communication_mode=mode,
            selection_basis="explicit_buffer_configuration",
            selection_profile=None,
            selection_reason="explicit modes and budgets",
            strategy_lifetime="buffer",
        )
        runtime_mode = mode
        if mode == "auto":
            homogeneous = all(record[1] == profile for record in records)
            name, capability, sms = profile if homogeneous else ("", (0, 0), 0)
            selection = select_rdma_strategy(
                hidden,
                communication_dtype,
                num_max_dispatch_tokens_per_rank,
                self.num_ranks,
                num_experts,
                name,
                capability,
                sms,
            )
            dispatch_mode = selection["dispatch"]["mode"]
            combine_mode = selection["combine"]["mode"]
            dispatch_num_sms = selection["dispatch"]["num_sms"]
            combine_num_sms = selection["combine"]["num_sms"]
            dispatch_num_contexts = selection["dispatch"]["num_contexts"]
            combine_num_contexts = selection["combine"]["num_contexts"]
            runtime_mode = "legacy"
            self._strategy_selection.update(
                selection_basis="collectively_agreed_capacity",
                selection_capacity=num_max_dispatch_tokens_per_rank,
                selection_profile=selection["profile"],
                selection_reason=selection["reason"]
                if homogeneous
                else "heterogeneous devices: legacy fallback",
            )
        error = None
        try:
            self._module = get_gin_module()
        except Exception as exc:
            error = _error(exc)
        self._raise_errors(self._gather(error), "loading JIT module")
        unique_id, error = None, None
        try:
            if self.rank == 0:
                unique_id = list(self._module.gin_get_unique_id())
        except Exception as exc:
            error = _error(exc)
        ids = self._gather((unique_id, error))
        self._raise_errors([item[1] for item in ids], "creating NCCL unique ID")
        error = None
        try:
            self._handle = _call(
                self._module.gin_create,
                self.rank,
                self.num_ranks,
                ids[0][0],
                num_max_dispatch_tokens_per_rank,
                hidden,
                num_experts,
                True,  # All remote traffic requires the GDAKI provider.
                -1 if traffic_class is None else traffic_class,
                runtime_mode,
                -1 if num_sms is None else num_sms,
                -1 if num_contexts is None else num_contexts,
                _PRECISION[communication_dtype],
                dispatch_mode or "",
                combine_mode or "",
                -1 if dispatch_num_sms is None else dispatch_num_sms,
                -1 if combine_num_sms is None else combine_num_sms,
                -1 if dispatch_num_contexts is None else dispatch_num_contexts,
                -1 if combine_num_contexts is None else combine_num_contexts,
                combine_reduction,
            )
        except Exception as exc:
            error = _error(exc)
        errors = self._gather(error)
        if any(errors):
            # Peers may have aborted during native initialization. Use the
            # local initialization-only cleanup, never collective destroy.
            cleanup_error = None
            if self._handle is not None:
                try:
                    _call(self._module.gin_abort, self._handle)
                    self._handle = None
                except Exception as exc:
                    cleanup_error = _error(exc)
            cleanup_errors = self._gather(cleanup_error)
            errors = [
                f"{failure or 'peer initialization failed'}; cleanup: {cleanup}"
                if cleanup
                else failure
                for failure, cleanup in zip(errors, cleanup_errors, strict=True)
            ]
            self._closed = True
            self._raise_errors(errors, "constructing NCCL GIN resources")

    def _gather(self, value):
        records = [None] * self.num_ranks
        dist.all_gather_object(records, value, group=self.group)
        return records

    def _raise_errors(self, errors, operation):
        failed = {rank: error for rank, error in enumerate(errors) if error}
        if failed:
            raise RuntimeError(
                f"RDMA MoE all-to-all failed collectively while {operation}: {failed}"
            )

    def _check_call(self):
        if self._closed or self._handle is None:
            raise RuntimeError("RDMA MoE all-to-all buffer is closed")
        if self._pending is not None:
            raise RuntimeError(
                "invoke the outstanding receive hook before another operation"
            )
        # Kernels mutate shared workspace state. The caller may change streams
        # only after establishing serial order with events or synchronization.
        # No device counters or metadata are copied back to enforce that order.
        return torch.cuda.current_stream(self.device).cuda_stream

    def _check_dimensions(self, capacity, num_experts):
        if capacity is not None and capacity != self.num_max_dispatch_tokens_per_rank:
            raise ValueError("dispatch capacity is fixed at buffer construction")
        if num_experts is not None and num_experts != self.num_experts:
            raise ValueError("expert count is fixed at buffer construction")

    def _tensor(self, tensor, name, dtype, shape):
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.device != self.device
            or tensor.dtype != dtype
            or not tensor.is_contiguous()
            or tuple(tensor.shape) != tuple(shape)
        ):
            raise ValueError(
                f"{name} must be contiguous {dtype} on {self.device} with shape {tuple(shape)}"
            )

    @staticmethod
    def _disjoint(outputs, inputs):
        def interval(tensor):
            return (
                tensor.data_ptr(),
                tensor.data_ptr() + tensor.numel() * tensor.element_size(),
            )

        for i, output in enumerate(outputs):
            if not output.numel():
                continue
            first, last = interval(output)
            for tensor in [*outputs[i + 1 :], *inputs]:
                if tensor is not None and tensor.numel():
                    begin, end = interval(tensor)
                    if first < end and begin < last:
                        raise ValueError(
                            "output tensors must not overlap inputs or other outputs"
                        )

    def _precision(self, precision):
        if precision != self.communication_dtype:
            raise ValueError("precision must match this buffer's communication_dtype")
        return _PRECISION[precision]

    def _launch(self, launch, return_recv_hook, stream):
        launch(1 if return_recv_hook else 3)
        self._slot ^= 1
        if not return_recv_hook:
            return None
        called = False
        marker = object()
        self._pending = marker

        def recv_hook():
            nonlocal called
            if called:
                raise RuntimeError("receive hook may only be invoked once")
            if self._closed or self._pending is not marker:
                raise RuntimeError("receive hook no longer owns a live operation")
            if torch.cuda.current_stream(self.device).cuda_stream != stream:
                raise RuntimeError(
                    "receive hook must run on the stream used for its send phase"
                )
            launch(2)
            called = True
            self._pending = None

        return recv_hook

    @flashinfer_api(trace=rdma_moe_dispatch_trace)
    def dispatch(
        self,
        x,
        topk_idx,
        *,
        x_scales=None,
        input_dtype="bf16",
        nvfp4_global_scale=256.0,
        topk_weights=None,
        out=None,
        cumulative_local_expert_recv_stats=None,
        return_recv_hook=False,
    ) -> DispatchResult:
        """Dispatch tokens into expert-major rows and return a :class:`DispatchResult`.

        ``input_dtype="prequantized"`` preserves the wire-format input contract.
        With ``input_dtype="bf16"``, ``x`` is BF16 ``[tokens, hidden]`` and
        ``x_scales`` must be None. FP8/NVFP4 inputs are dynamically block
        quantized on the current stream before dispatch; BF16 is sent directly.
        NVFP4 uses the positive finite host scalar ``nvfp4_global_scale``.
        All ranks and receiving experts must externally agree on that value:
        dispatch transmits block scales, but not the global scale.

        ``out`` reuses recv_x, recv_scales, counts, src_info and ranges.
        BF16 recv_scales has
        shape ``[local_experts, world_size * capacity, 0]`` and uint8 dtype.
        Routing must contain distinct expert ids per token, with -1 for unused
        picks. Value checks are the caller's responsibility and do not run here.

        ``bf16_local_partial_planned`` requires float32 ``topk_weights`` with
        the same shape as ``topk_idx``. Dispatch snapshots routes and weights,
        transmits them with the tokens, and builds a handle-owned combine plan.
        That work is part of dispatch. Consume it with ``combine(x, handle)``;
        later changes to caller routes/weights do not alter the handle's plan.
        Other reduction modes reject ``topk_weights`` here and take weights
        at combine time as before. Input tensors must remain valid and unchanged
        until this asynchronous dispatch completes.
        """
        stream = self._check_call()
        precision = self.communication_dtype
        precision_id = self._precision(precision)
        if input_dtype not in ("prequantized", "bf16"):
            raise ValueError("input_dtype must be 'prequantized' or 'bf16'")
        if not isinstance(x, torch.Tensor) or x.ndim != 2:
            raise ValueError("x must be a two-dimensional token tensor")
        tokens = x.shape[0]
        width = self.hidden // 2 if precision == "nvfp4" else self.hidden
        scale_width = {
            "bf16": 0,
            "fp8": self.hidden // 128,
            "nvfp4": self.hidden // 16,
        }[precision]
        dtype = torch.bfloat16 if precision == "bf16" else torch.uint8
        scale_dtype = torch.float32 if precision == "fp8" else torch.uint8
        if input_dtype == "bf16":
            self._tensor(x, "x", torch.bfloat16, (tokens, self.hidden))
            if x_scales is not None:
                raise ValueError("BF16 input dispatch requires x_scales=None")
            if precision == "nvfp4":
                nvfp4_global_scale = self._uniform_nvfp4_scale(nvfp4_global_scale)
        else:
            self._tensor(x, "x", dtype, (tokens, width))
        if tokens > self.num_max_dispatch_tokens_per_rank:
            raise ValueError("token count exceeds the fixed dispatch capacity")
        if precision == "bf16":
            if x_scales is not None:
                raise ValueError("BF16 dispatch requires x_scales=None")
        elif input_dtype == "prequantized":
            self._tensor(x_scales, "x_scales", scale_dtype, (tokens, scale_width))
        if (
            not isinstance(topk_idx, torch.Tensor)
            or topk_idx.ndim != 2
            or not 1 <= topk_idx.shape[1] <= 8
        ):
            raise ValueError(
                "topk_idx must have shape [tokens, top_k] with top_k in [1,8]"
            )
        self._tensor(topk_idx, "topk_idx", torch.int32, (tokens, topk_idx.shape[1]))
        planned = self.combine_reduction == _PLANNED_REDUCTION
        if planned:
            self._tensor(topk_weights, "topk_weights", torch.float32, topk_idx.shape)
        elif topk_weights is not None:
            raise ValueError(
                "dispatch topk_weights requires bf16_local_partial_planned"
            )
        local_experts = self.num_experts // self.num_ranks
        cap = self.num_ranks * self.num_max_dispatch_tokens_per_rank
        stats = cumulative_local_expert_recv_stats
        if stats is not None:
            self._tensor(stats, "recv_stats", torch.int32, (local_experts,))
        shapes = [
            (local_experts, cap, width),
            (local_experts, cap, scale_width),
            (local_experts,),
            (local_experts, cap),
            (local_experts, self.num_ranks),
        ]
        dtypes = [dtype, scale_dtype, torch.int32, torch.int32, torch.int64]
        if out is None:
            outputs = [
                torch.empty(shape, dtype=dt, device=self.device)
                for shape, dt in zip(shapes, dtypes, strict=True)
            ]
        else:
            if not isinstance(out, (tuple, list)) or len(out) != 5:
                raise ValueError("dispatch out must contain five tensors")
            outputs = list(out)
            for i, (tensor, shape, dt) in enumerate(
                zip(outputs, shapes, dtypes, strict=True)
            ):
                self._tensor(tensor, f"out[{i}]", dt, shape)
            self._disjoint(outputs, [x, x_scales, topk_idx, topk_weights, stats])
        recv_x, recv_scales, counts, src, ranges = outputs[:5]
        source_x = x
        if input_dtype == "bf16":
            # Reject invalid output metadata before submitting any quantization.
            for name, tensor in (("x", x), ("out[0]", recv_x), ("out[1]", recv_scales)):
                if tensor.numel() and tensor.data_ptr() % 16:
                    raise ValueError(f"{name} must have a 16-byte aligned data pointer")
            if precision != "bf16":
                x = torch.empty((tokens, width), dtype=dtype, device=self.device)
                x_scales = torch.empty(
                    (tokens, scale_width), dtype=scale_dtype, device=self.device
                )
                quantization = get_quantization_module()
                if precision == "fp8":
                    _call(quantization.quantize_fp8, source_x, x, x_scales)
                else:
                    _call(
                        quantization.quantize_nvfp4_uniform,
                        source_x,
                        nvfp4_global_scale,
                        x,
                        x_scales,
                    )
        args = [self._handle, precision_id, x, x_scales, topk_idx, stats, *outputs]
        slot = self._slot
        function = self._module.gin_dispatch
        plan = None
        if planned:
            plan = _CombinePlan(self, tokens, topk_idx.shape[1])
            args.extend([topk_weights, *plan.tensors])
            function = self._module.gin_dispatch_planned

        # A pending receive hook owns both the quantized send buffers in args
        # and the original BF16 source. Quantization runs only before send.
        def launch(phases, _source=source_x):
            _call(function, *args, phases, slot)

        hook = self._launch(launch, return_recv_hook, stream)
        values = (
            src,
            ranges,
            self.num_max_dispatch_tokens_per_rank,
            self.hidden,
            self.num_experts,
        )
        handle = DispatchHandle(
            values, owner=self._owner, precision=precision, combine_plan=plan
        )
        return DispatchResult(recv_x, recv_scales, counts, handle, hook)

    @flashinfer_api(trace=rdma_moe_combine_trace)
    def combine(
        self,
        x,
        handle,
        *,
        topk_idx=None,
        topk_weights=None,
        global_scale=None,
        out=None,
        return_recv_hook=False,
    ) -> CombineResult:
        """Combine BF16 expert rows into token order using the fixed reduction.

        ``fp32`` accumulates weighted rows in FP32 and rounds once to BF16.
        ``bf16_local_partial`` rounds weighted leaves to BF16, accumulates each
        owner's leaves in route order in FP32, rounds owner partials to BF16,
        then sums owners in first-appearance order in FP32 before the final cast.
        These modes consume the supplied ``topk_idx`` and ``topk_weights`` at
        combine time. NVFP4 also requires float32 ``global_scale`` of shape
        ``[local_experts, world_size * capacity, 1]`` for expert output quantization.

        ``bf16_local_partial_planned`` uses the same staged rounding and the
        route/weight snapshot owned by ``handle`` from dispatch. In this mode,
        omit ``topk_idx``, ``topk_weights`` and ``global_scale``. Changing the
        snapshot requires a new dispatch; a captured dispatch rebuilds it on
        each replay. Older handles keep their plans after later dispatches,
        provided their paired expert outputs have not been overwritten.

        ``out`` optionally reuses the BF16 ``[tokens, hidden]`` result tensor.
        A receive hook, when requested, must execute once on the send stream
        before the output is consumed or another operation is submitted.
        """
        if self.combine_reduction == _PLANNED_REDUCTION:
            if (
                topk_idx is not None
                or topk_weights is not None
                or global_scale is not None
            ):
                raise ValueError(
                    "planned combine uses the dispatch snapshot; omit routes, weights and global_scale"
                )
            return self._combine_planned(
                x, handle, return_recv_hook=return_recv_hook, out=out
            )
        stream = self._check_call()
        precision = self.communication_dtype
        precision_id = self._precision(precision)
        if (
            not isinstance(handle, DispatchHandle)
            or handle._owner is not self._owner
            or handle.precision != precision
        ):
            raise ValueError(
                "combine requires a dispatch handle produced by this buffer and precision"
            )
        src, ranges, capacity, hidden, experts = handle[:5]
        self._check_dimensions(capacity, experts)
        if hidden != self.hidden or len(handle) != 5:
            raise ValueError(
                "dispatch handle dimensions/backend do not match this buffer"
            )
        local_experts = self.num_experts // self.num_ranks
        cap = self.num_ranks * capacity
        self._tensor(x, "x", torch.bfloat16, (local_experts, cap, hidden))
        if (
            not isinstance(topk_idx, torch.Tensor)
            or topk_idx.ndim != 2
            or not 1 <= topk_idx.shape[1] <= 8
            or topk_idx.shape[0] > capacity
        ):
            raise ValueError(
                "combine topk_idx must have shape [tokens<=capacity, top_k in 1..8]"
            )
        self._tensor(topk_idx, "topk_idx", torch.int32, topk_idx.shape)
        self._tensor(topk_weights, "topk_weights", torch.float32, topk_idx.shape)
        self._tensor(src, "src_info", torch.int32, (local_experts, cap))
        self._tensor(ranges, "ranges", torch.int64, (local_experts, self.num_ranks))
        if precision == "nvfp4" and global_scale is None:
            raise ValueError("NVFP4 combine requires global_scale")
        if global_scale is not None:
            self._tensor(
                global_scale, "global_scale", torch.float32, (local_experts, cap, 1)
            )
        metadata = [src, ranges]
        shape = (topk_idx.shape[0], hidden)
        output = (
            torch.empty(shape, dtype=torch.bfloat16, device=self.device)
            if out is None
            else out
        )
        self._tensor(output, "out", torch.bfloat16, shape)
        self._disjoint([output], [x, global_scale, topk_idx, topk_weights, *metadata])
        args = [
            self._handle,
            precision_id,
            x,
            global_scale,
            topk_idx,
            topk_weights,
            *metadata,
            output,
        ]
        slot = self._slot
        function = self._module.gin_combine

        def launch(phases):
            _call(function, *args, phases, slot)

        hook = self._launch(launch, return_recv_hook, stream)
        return CombineResult(output, hook)

    def _combine_planned(
        self,
        x,
        handle,
        return_recv_hook=False,
        out=None,
    ):
        """Combine with the route/weight snapshot created by dispatch.

        Available only for ``bf16_local_partial_planned`` buffers. Weighted
        BF16 leaves, ordered local FP32 sums, BF16 owner partials and final
        first-owner-order FP32 reduction use the local-partial rounding contract.
        Metadata exchange and plan construction execute in dispatch. Repeating
        combine uses that fixed plan; changing routes or weights requires a new
        dispatch. Replaying a graph containing dispatch rebuilds its plan.

        The handle retains its private plan, independently of later dispatches.
        Reusing dispatch output tensors still overwrites their expert rows and
        metadata; the caller must preserve the expert output paired with this
        handle. Execution and receive-hook ordering follow the normal buffer
        contract.
        """
        stream = self._check_call()
        if self.combine_reduction != _PLANNED_REDUCTION:
            raise ValueError("planned combine requires bf16_local_partial_planned")
        if (
            not isinstance(handle, DispatchHandle)
            or handle._owner is not self._owner
            or handle.precision != "bf16"
            or len(handle) != 5
            or not isinstance(handle._combine_plan, _CombinePlan)
            or handle._combine_plan.owner is not self._owner
        ):
            raise ValueError("planned combine requires this buffer's planned handle")
        plan = handle._combine_plan
        src, ranges, capacity, hidden, experts = handle
        self._check_dimensions(capacity, experts)
        if hidden != self.hidden:
            raise ValueError("dispatch handle hidden dimension does not match")
        local_experts = self.num_experts // self.num_ranks
        rows = self.num_ranks * capacity
        self._tensor(x, "x", torch.bfloat16, (local_experts, rows, hidden))
        self._tensor(src, "src_info", torch.int32, (local_experts, rows))
        self._tensor(ranges, "ranges", torch.int64, (local_experts, self.num_ranks))
        shape = (plan.tokens, hidden)
        output = (
            torch.empty(shape, dtype=torch.bfloat16, device=self.device)
            if out is None
            else out
        )
        self._tensor(output, "out", torch.bfloat16, shape)
        self._disjoint([output], [x, src, ranges, *plan.tensors])
        args = [self._handle, x, src, ranges, output, *plan.tensors]
        slot = self._slot

        def launch(phases):
            _call(self._module.gin_combine_planned, *args, phases, slot)

        hook = self._launch(launch, return_recv_hook, stream)
        return CombineResult(output, hook)

    @staticmethod
    def _uniform_nvfp4_scale(value):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("nvfp4_global_scale must be a positive finite host scalar")
        try:
            # Validate the actual kernel argument without creating a CUDA
            # tensor, synchronizing, or adding a fill kernel during capture.
            value = struct.unpack("f", struct.pack("f", float(value)))[0]
        except (OverflowError, struct.error) as exc:
            raise ValueError(
                "nvfp4_global_scale must be positive and finite in float32"
            ) from exc
        if not math.isfinite(value) or value <= 0:
            raise ValueError(
                "nvfp4_global_scale must be positive and finite in float32"
            )
        return value

    @staticmethod
    def _quant_input(x, dtype, hidden=None):
        if (
            not isinstance(x, torch.Tensor)
            or not x.is_cuda
            or not x.is_contiguous()
            or x.dtype != dtype
            or x.ndim not in (2, 3)
        ):
            raise ValueError(
                "quantization input must be contiguous CUDA rank 2/3 with the expected dtype"
            )
        width = x.shape[-1] if hidden is None else hidden
        if width not in _HIDDEN:
            raise ValueError(f"quantization hidden must be one of {_HIDDEN}")
        if x.numel() // x.shape[-1] >= 2**31:
            raise ValueError("quantization row count exceeds int32")

    @staticmethod
    def _global_scale(x, scale):
        if (
            not isinstance(scale, torch.Tensor)
            or scale.device != x.device
            or scale.dtype != torch.float32
            or not scale.is_contiguous()
            or tuple(scale.shape) != (*x.shape[:-1], 1)
        ):
            raise ValueError(
                "global_scale must be contiguous float32 on the input device with shape [...,1]"
            )

    @staticmethod
    def quantize_bf16_to_nvfp4(x, global_scale):
        RdmaMoeAlltoAll._quant_input(x, torch.bfloat16)
        RdmaMoeAlltoAll._global_scale(x, global_scale)
        packed = torch.empty(
            (*x.shape[:-1], x.shape[-1] // 2), dtype=torch.uint8, device=x.device
        )
        scales = torch.empty(
            (*x.shape[:-1], x.shape[-1] // 16), dtype=torch.uint8, device=x.device
        )
        _call(get_quantization_module().quantize_nvfp4, x, global_scale, packed, scales)
        return packed, scales

    @staticmethod
    def dequantize_nvfp4_to_bf16(packed_nvfp4_x, global_scale, fp8_scales):
        packed = packed_nvfp4_x
        RdmaMoeAlltoAll._quant_input(packed, torch.uint8, packed.shape[-1] * 2)
        RdmaMoeAlltoAll._global_scale(packed, global_scale)
        hidden = packed.shape[-1] * 2
        shape = (*packed.shape[:-1], hidden // 16)
        if (
            not isinstance(fp8_scales, torch.Tensor)
            or fp8_scales.device != packed.device
            or fp8_scales.dtype != torch.uint8
            or not fp8_scales.is_contiguous()
            or tuple(fp8_scales.shape) != shape
        ):
            raise ValueError(
                "NVFP4 scales must be contiguous uint8 with shape [...,hidden/16]"
            )
        output = torch.empty(
            (*packed.shape[:-1], hidden), dtype=torch.bfloat16, device=packed.device
        )
        _call(
            get_quantization_module().dequantize_nvfp4,
            packed,
            global_scale,
            fp8_scales,
            output,
        )
        return output

    @staticmethod
    def quantize_bf16_to_fp8(x):
        RdmaMoeAlltoAll._quant_input(x, torch.bfloat16)
        packed = torch.empty(x.shape, dtype=torch.uint8, device=x.device)
        scales = torch.empty(
            (*x.shape[:-1], x.shape[-1] // 128), dtype=torch.float32, device=x.device
        )
        _call(get_quantization_module().quantize_fp8, x, packed, scales)
        return packed, scales

    @staticmethod
    def dequantize_fp8_to_bf16(packed_fp8_x, scales):
        packed = packed_fp8_x
        RdmaMoeAlltoAll._quant_input(packed, torch.uint8)
        if (
            not isinstance(scales, torch.Tensor)
            or scales.device != packed.device
            or scales.dtype != torch.float32
            or not scales.is_contiguous()
            or tuple(scales.shape) != (*packed.shape[:-1], packed.shape[-1] // 128)
        ):
            raise ValueError(
                "FP8 scales must be contiguous float32 with shape [...,hidden/128]"
            )
        output = torch.empty(packed.shape, dtype=torch.bfloat16, device=packed.device)
        _call(get_quantization_module().dequantize_fp8, packed, scales, output)
        return output

    def backend_info(self):
        if self._closed:
            raise RuntimeError("RDMA MoE all-to-all buffer is closed")
        return {
            **_plain(self._module.gin_info(self._handle)),
            **self._strategy_selection,
        }

    def close(self):
        """Collectively retire resources after graphs and GPU users finish."""
        if self._closed:
            return
        error = None
        try:
            if self._pending is not None:
                raise RuntimeError("invoke the outstanding receive hook before closing")
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("close must happen outside CUDA graph capture")
            torch.cuda.synchronize(self.device)
        except Exception as exc:
            error = _error(exc)
        self._raise_errors(self._gather(error), "preparing close")
        function = self._module.gin_close
        error = None
        try:
            if self._handle is not None:
                _call(function, self._handle)
                self._handle = None
        except Exception as exc:
            error = _error(exc)
        # Ranks that already released their resources still participate in a
        # later collective close retry if a peer's release failed. Marking
        # them closed before agreement would make that retry hang.
        self._raise_errors(self._gather(error), "closing native resources")
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
