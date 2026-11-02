"""Metadata lowering and resolved launch ownership for PLE residual state."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import torch
import triton

from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.preparation import FrozenMapping, MemoryRequirements, Plan
from ._contracts import LayerCaps, _bind_layer, _materialize_layout
from ._tuning import OPERANDS, PleConfig, PleQuery, TUNING


_BOOL = frozenset(("state_is_fresh", "request_is_prefill"))
_I32 = frozenset(
    ("query_start_loc", "num_accepted_tokens", "num_seqs", "num_tokens", "request_ids")
)
_SCRATCH = frozenset(("normalized_u", "gathered_state", "request_ids"))


def _alignment(tensor):
    pointer = tensor.data_ptr()
    return min(16, pointer & -pointer) if pointer else 16


def invocation_from_tensors(*, conv_state, scratch=None, **tensors):
    tensors = {**tensors, "conv_state": conv_state}
    if tensors.get("request_is_prefill") is None:
        tensors["request_is_prefill"] = tensors["state_is_fresh"]
    missing = set(OPERANDS) - _SCRATCH - tensors.keys()
    if missing:
        raise ValueError(
            f"PLE invocation is missing tensors: {', '.join(sorted(missing))}"
        )
    scratch_alignment = 16
    if scratch is not None:
        if isinstance(scratch, Mapping):
            scratch = scratch["ple_layer"]
        elif isinstance(scratch, Sequence) and not isinstance(scratch, torch.Tensor):
            if len(scratch) != 1:
                raise ValueError("PLE residual uses one scratch buffer")
            scratch = scratch[0]
        scratch_alignment = _alignment(scratch)
    return FrozenMapping(
        {
            "state_strides": tuple(conv_state.stride()),
            "input_alignments": tuple(
                _alignment(tensors[name]) if name not in _SCRATCH else scratch_alignment
                for name in OPERANDS
            ),
        }
    )


@dataclass(frozen=True)
class _CompilePointer:
    dtype: torch.dtype
    alignment: int

    def data_ptr(self):
        return self.alignment


def _geometry(query):
    channels = query.streams * query.hidden_size
    length = query.dilation * (query.kernel_size - 1)
    block_h = max(16, triton.next_power_of_2(query.hidden_size))
    return channels, length, length + query.max_speculative_tokens, block_h


@program_cache(scope="preparation")
def compile_layer(query_payload, config_payload, ordinal):
    from . import _kernels as kernels

    query = PleQuery(**dict(query_payload))
    TUNING.validate_config(
        query, PleConfig.from_config(FrozenMapping(config_payload)), None
    )
    p = {}
    for name, alignment in zip(OPERANDS, query.input_alignments):
        dtype = (
            torch.bool
            if name in _BOOL
            else torch.int32
            if name in _I32
            else torch.int64
            if name == "state_slot_ids"
            else torch.bfloat16
        )
        p[name] = _CompilePointer(dtype, alignment)
    channels, length, capacity, block_h = _geometry(query)
    decode, mixed = query.mode == "decode", query.mode == "mixed"
    m, n = query.max_tokens, query.max_seqs
    channel_grid = triton.cdiv(channels, kernels._CHANNEL_BLOCK)
    state_options = dict(
        CHANNELS=channels,
        STATE_LENGTH=length,
        STATE_CAPACITY=capacity,
        stride_state_slot=query.state_strides[0],
        stride_state_channel=query.state_strides[1],
        stride_state_position=query.state_strides[2],
        MAX_SPECULATIVE=query.max_speculative_tokens,
        DECODE=decode,
        MIXED=mixed,
        BLOCK_C=kernels._CHANNEL_BLOCK,
        num_warps=4,
        grid=(n, channel_grid),
    )
    with torch.cuda.device(ordinal):
        return (
            kernels._request_ids_kernel.warmup(
                p["query_start_loc"],
                p["num_seqs"],
                p["num_tokens"],
                p["request_ids"],
                MAX_TOKENS=m,
                num_warps=1,
                grid=(m,),
            ),
            kernels._prepare_history_kernel.warmup(
                p["conv_state"],
                p["state_slot_ids"],
                p["state_is_fresh"],
                p["num_accepted_tokens"],
                p["request_is_prefill"],
                p["num_seqs"],
                p["gathered_state"],
                **state_options,
            ),
            kernels._gated_u_norm_kernel.warmup(
                p["residual"],
                p["key"],
                p["value"],
                p["k_norm_weight"],
                p["q_norm_weight"],
                p["u_norm_weight"],
                p["request_ids"],
                p["state_slot_ids"],
                p["num_tokens"],
                p["out"],
                p["normalized_u"],
                1e-6,
                STREAMS=query.streams,
                HIDDEN_SIZE=query.hidden_size,
                CHANNELS=channels,
                BLOCK_H=block_h,
                num_warps=8 if block_h >= 2048 else 4,
                grid=(m, query.streams),
            ),
            kernels._dilated_conv_kernel.warmup(
                p["normalized_u"],
                p["gathered_state"],
                p["conv_weight"],
                p["query_start_loc"],
                p["request_ids"],
                p["state_slot_ids"],
                p["num_tokens"],
                p["out"],
                CHANNELS=channels,
                STATE_LENGTH=length,
                KERNEL_SIZE=query.kernel_size,
                DILATION=query.dilation,
                BLOCK_C=kernels._CHANNEL_BLOCK,
                num_warps=4,
                grid=(m, channel_grid),
            ),
            kernels._update_state_kernel.warmup(
                p["normalized_u"],
                p["gathered_state"],
                p["query_start_loc"],
                p["state_slot_ids"],
                p["request_is_prefill"],
                p["num_seqs"],
                p["conv_state"],
                **state_options,
            ),
        )


def _binding_tensors(binding, token_count=None):
    residual = (
        binding.residual if token_count is None else binding.residual[:token_count]
    )
    flags = (
        binding.request_is_prefill
        if binding.request_is_prefill is not None
        else binding.state_is_fresh
    )
    return (
        residual,
        binding.key,
        binding.value,
        binding.k_norm_weight,
        binding.q_norm_weight,
        binding.u_norm_weight,
        binding.conv_weight,
        binding.query_start_loc,
        binding.state_slot_ids,
        binding.state_is_fresh,
        binding.num_accepted_tokens,
        flags,
        binding.num_seqs,
        binding.num_tokens,
        binding.conv_state,
        binding.out,
        binding.normalized_u,
        binding.gathered_state,
        binding.request_ids,
    )


@dataclass(frozen=True)
class _PleState:
    query: PleQuery
    layout: object
    programs: tuple
    channels: int = field(init=False)
    state_length: int = field(init=False)
    state_capacity: int = field(init=False)
    block_h: int = field(init=False)
    channel_grid: int = field(init=False)
    decode: bool = field(init=False)
    mixed: bool = field(init=False)
    state_args: tuple = field(init=False)

    def __post_init__(self):
        channels, length, capacity, block_h = _geometry(self.query)
        for name, value in (
            ("channels", channels),
            ("state_length", length),
            ("state_capacity", capacity),
            ("block_h", block_h),
            ("channel_grid", triton.cdiv(channels, 128)),
            ("decode", self.query.mode == "decode"),
            ("mixed", self.query.mode == "mixed"),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "state_args",
            (
                channels,
                length,
                capacity,
                *self.query.state_strides,
                self.query.max_speculative_tokens,
                self.decode,
                self.mixed,
                128,
            ),
        )

    def _check(self, tensors):
        for name, tensor, alignment in zip(
            OPERANDS, tensors, self.query.input_alignments
        ):
            if tensor.device != self.layout.caps.device:
                raise ValueError(f"PLE {name} device differs from preparation")
            if tensor.numel() and _alignment(tensor) != alignment:
                raise ValueError(f"PLE {name} alignment differs from preparation")
        if tuple(tensors[14].stride()) != self.query.state_strides:
            raise ValueError("PLE convolution state strides differ from preparation")

    def bind(self, **kwargs):
        binding = _bind_layer(self.layout, **kwargs)
        self._check(_binding_tensors(binding))
        return binding

    def run(self, binding, *, eps, token_count=None):
        if binding._state is not self.layout:
            raise ValueError("binding belongs to another PLE plan")
        if token_count is None:
            token_count = self.query.max_tokens
        if not 0 <= token_count <= self.query.max_tokens:
            raise ValueError("PLE token count exceeds prepared capacity")
        if not self.mixed and token_count != self.query.max_tokens:
            raise ValueError("only mixed PLE exposes a host token-count bound")
        self.run_tensors(*_binding_tensors(binding, token_count), eps=eps)
        return binding.out[:token_count]

    def run_tensors(
        self,
        residual,
        key,
        value,
        k_norm_weight,
        q_norm_weight,
        u_norm_weight,
        conv_weight,
        query_start_loc,
        state_slot_ids,
        state_is_fresh,
        num_accepted_tokens,
        request_is_prefill,
        num_seqs,
        num_tokens,
        conv_state,
        out,
        normalized_u,
        gathered_state,
        request_ids,
        *,
        eps,
    ):
        eps = float(eps)
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("PLE epsilon must be finite and positive")
        self._check(
            (
                residual,
                key,
                value,
                k_norm_weight,
                q_norm_weight,
                u_norm_weight,
                conv_weight,
                query_start_loc,
                state_slot_ids,
                state_is_fresh,
                num_accepted_tokens,
                request_is_prefill,
                num_seqs,
                num_tokens,
                conv_state,
                out,
                normalized_u,
                gathered_state,
                request_ids,
            )
        )
        q, p = self.query, self.programs
        count = int(residual.shape[0])
        state_args = self.state_args
        with torch.cuda.device(self.layout.caps.device):
            p[0][(count, 1, 1)](
                query_start_loc, num_seqs, num_tokens, request_ids, q.max_tokens
            )
            p[1][(q.max_seqs, self.channel_grid, 1)](
                conv_state,
                state_slot_ids,
                state_is_fresh,
                num_accepted_tokens,
                request_is_prefill,
                num_seqs,
                gathered_state,
                *state_args,
            )
            p[2][(count, q.streams, 1)](
                residual,
                key,
                value,
                k_norm_weight,
                q_norm_weight,
                u_norm_weight,
                request_ids,
                state_slot_ids,
                num_tokens,
                out,
                normalized_u,
                eps,
                q.streams,
                q.hidden_size,
                self.channels,
                self.block_h,
            )
            p[3][(count, self.channel_grid, 1)](
                normalized_u,
                gathered_state,
                conv_weight,
                query_start_loc,
                request_ids,
                state_slot_ids,
                num_tokens,
                out,
                self.channels,
                self.state_length,
                q.kernel_size,
                q.dilation,
                128,
            )
            p[4][(q.max_seqs, self.channel_grid, 1)](
                normalized_u,
                gathered_state,
                query_start_loc,
                state_slot_ids,
                request_is_prefill,
                num_seqs,
                conv_state,
                *state_args,
            )


def make_plan(caps: LayerCaps, *, invocation, override):
    invocation = FrozenMapping(invocation)
    if set(invocation) - {"state_strides", "input_alignments"}:
        raise ValueError("unknown PLE residual invocation metadata")
    query = PleQuery(
        mode=caps.mode,
        dtype=str(caps.dtype).removeprefix("torch."),
        max_tokens=caps.max_tokens,
        max_seqs=caps.max_seqs,
        max_state_slots=caps.max_state_slots,
        max_speculative_tokens=caps.max_speculative_tokens,
        streams=caps.streams,
        hidden_size=caps.hidden_size,
        kernel_size=caps.kernel_size,
        dilation=caps.dilation,
        **dict(invocation),
    )

    # The compile factory receives the full declared query; the selection
    # key omits the pool's state slot count.
    query_payload = {
        name: getattr(query, name) for name in PleQuery.__dataclass_fields__
    }

    def compile_jobs(config, device):
        return (
            CompileJob.create(
                "b12x.sequence.ple._preparation:compile_layer",
                query_payload,
                config.to_dict(),
                device.ordinal,
            ),
        )

    def memory(config, device):
        return MemoryRequirements(
            scratch=_materialize_layout(caps, config).scratch_specs()
        )

    def materialize(selection, device):
        return _PleState(
            query,
            _materialize_layout(caps, selection.config),
            compile_layer(query_payload, selection.config.to_dict(), device.ordinal),
        )

    return Plan(
        contract=TUNING,
        query=query,
        invocation=invocation,
        override=override,
        shared=False,
        _compile_jobs=compile_jobs,
        _memory_requirements=memory,
        _materialize=materialize,
        _device=caps.device,
    )
