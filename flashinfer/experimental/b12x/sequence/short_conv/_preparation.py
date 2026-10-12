"""Metadata compilation and retained raw-convolution pipeline launches."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import triton
from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x._lib.scratch import scratch_buffer_spec, scratch_tensor
from b12x._lib.scratch_layout import (
    SCRATCH_ALIGN_BYTES,
    align_up,
    materialize_scratch_view,
)
from b12x.preparation import FrozenMapping, MemoryRequirements, Plan

from ._tuning import OPERANDS, TUNING, ShortConvQuery

_SCRATCH = frozenset(("gathered_state", "request_ids"))
_BOOL = frozenset(("state_is_fresh", "request_is_prefill"))
_I32 = frozenset(
    ("query_start_loc", "num_accepted_tokens", "num_seqs", "num_tokens", "request_ids")
)


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
            f"short convolution invocation is missing {', '.join(sorted(missing))}"
        )
    alignment = 16
    if scratch is not None:
        if isinstance(scratch, Mapping):
            scratch = scratch["short_conv"]
        elif isinstance(scratch, Sequence) and not isinstance(scratch, torch.Tensor):
            if len(scratch) != 1:
                raise ValueError("short convolution requires one scratch buffer")
            scratch = scratch[0]
        alignment = _alignment(scratch)
    return FrozenMapping(
        {
            "state_strides": tuple(conv_state.stride()),
            "input_row_stride": tensors["x"].stride(0),
            "input_alignments": tuple(
                alignment if name in _SCRATCH else _alignment(tensors[name])
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


def _state_args(q):
    return (
        q.channels,
        q.state_length,
        q.state_capacity,
        *q.state_strides,
        q.max_speculative_tokens,
        q.mode == "decode",
        q.mode == "mixed",
        128,
    )


def _speculative_args(q):
    return (
        q.channels,
        q.state_length,
        q.state_capacity,
        *q.state_strides,
        q.max_speculative_tokens,
        q.kernel_size,
        q.dilation,
        q.input_row_stride,
        q.max_seqs,
        q.max_tokens,
        128,
    )


def _token_block(q) -> int:
    """Tokens per convolution program: tiles amortize row loads in prefill."""
    return 32 if q.max_tokens >= 256 else 1


@program_cache(scope="preparation")
def compile_pipeline(payload, ordinal):
    from .. import _short_conv as kernels

    q = ShortConvQuery(**dict(payload))
    p = {
        name: _CompilePointer(
            torch.bool
            if name in _BOOL
            else torch.int32
            if name in _I32
            else torch.int64
            if name == "state_slot_ids"
            else torch.bfloat16,
            alignment,
        )
        for name, alignment in zip(OPERANDS, q.input_alignments)
    }
    channel_grid = triton.cdiv(q.channels, 128)
    token_tiles = triton.cdiv(q.max_tokens, _token_block(q))
    with torch.cuda.device(ordinal):
        if q.mode == "decode" and q.max_speculative_tokens > 0:
            return (
                kernels._speculative_decode_short_conv_kernel.warmup(
                    *(
                        p[name]
                        for name in (
                            "x",
                            "conv_weight",
                            "query_start_loc",
                            "state_slot_ids",
                            "state_is_fresh",
                            "num_accepted_tokens",
                            "num_seqs",
                            "num_tokens",
                            "conv_state",
                            "out",
                        )
                    ),
                    *_speculative_args(q),
                    num_warps=4,
                    grid=(q.max_seqs + 1, channel_grid),
                ),
            )
        if q.mode == "decode" and q.max_speculative_tokens == 0:
            return (
                kernels._decode_short_conv_kernel.warmup(
                    *(
                        p[name]
                        for name in (
                            "x",
                            "conv_weight",
                            "query_start_loc",
                            "state_slot_ids",
                            "state_is_fresh",
                            "num_seqs",
                            "num_tokens",
                            "conv_state",
                            "out",
                        )
                    ),
                    q.channels,
                    q.kernel_size,
                    q.dilation,
                    q.input_row_stride,
                    q.state_strides[0],
                    128,
                    num_warps=4,
                    grid=(q.max_tokens, channel_grid),
                ),
            )
        return (
            kernels._request_ids_kernel.warmup(
                p["query_start_loc"],
                p["num_seqs"],
                p["num_tokens"],
                p["request_ids"],
                q.max_tokens,
                num_warps=1,
                grid=(q.max_tokens,),
            ),
            kernels._prepare_history_kernel.warmup(
                p["conv_state"],
                p["state_slot_ids"],
                p["state_is_fresh"],
                p["num_accepted_tokens"],
                p["request_is_prefill"],
                p["num_seqs"],
                p["gathered_state"],
                *_state_args(q),
                num_warps=4,
                grid=(q.max_seqs, channel_grid),
            ),
            kernels._short_conv_tiled_kernel.warmup(
                p["x"],
                p["gathered_state"],
                p["conv_weight"],
                p["query_start_loc"],
                p["request_ids"],
                p["state_slot_ids"],
                p["num_tokens"],
                p["out"],
                q.channels,
                q.state_length,
                q.kernel_size,
                q.dilation,
                128,
                q.input_row_stride,
                _token_block(q),
                q.max_tokens,
                num_warps=4,
                grid=(token_tiles, channel_grid),
            )
            if _token_block(q) > 1
            else kernels._short_conv_kernel.warmup(
                p["x"],
                p["gathered_state"],
                p["conv_weight"],
                p["query_start_loc"],
                p["request_ids"],
                p["state_slot_ids"],
                p["num_tokens"],
                p["out"],
                q.channels,
                q.state_length,
                q.kernel_size,
                q.dilation,
                128,
                False,
                q.input_row_stride,
                num_warps=4,
                grid=(q.max_tokens, channel_grid),
            ),
            kernels._update_state_kernel.warmup(
                p["x"],
                p["gathered_state"],
                p["query_start_loc"],
                p["state_slot_ids"],
                p["request_is_prefill"],
                p["num_seqs"],
                p["conv_state"],
                *_state_args(q),
                q.input_row_stride,
                num_warps=4,
                grid=(q.max_seqs, channel_grid),
            ),
        )


def _scratch_layout(query, device):
    ids_offset = align_up(
        query.max_seqs * query.channels * query.state_length * 2,
        SCRATCH_ALIGN_BYTES,
    )
    spec = scratch_buffer_spec(
        "short_conv",
        nbytes=ids_offset + query.max_tokens * 4,
        device=device,
    )
    return ids_offset, spec


@dataclass(frozen=True)
class PreparedShortConv:
    query: ShortConvQuery
    device: torch.device
    programs: tuple

    def _check(self, tensors):
        for name, tensor, alignment in zip(
            OPERANDS, tensors, self.query.input_alignments
        ):
            if tensor.device != self.device or _alignment(tensor) != alignment:
                raise ValueError(
                    f"short convolution {name} device/alignment differs from preparation"
                )
        if tensors[0].stride() != (self.query.input_row_stride, 1):
            raise ValueError("short convolution input strides differ from preparation")
        if tuple(tensors[9].stride()) != self.query.state_strides:
            raise ValueError(
                "short convolution history strides differ from preparation"
            )

    def bind(
        self,
        *,
        plan,
        scratch,
        x,
        conv_weight,
        query_start_loc,
        state_slot_ids,
        state_is_fresh,
        num_accepted_tokens,
        num_seqs,
        num_tokens,
        conv_state,
        out,
        request_is_prefill=None,
    ):
        from ..ple._contracts import (
            _require_conv_state_tensor,
            _require_mutation_alias_contract,
            _require_tensor,
        )
        from .api import Binding

        q = self.query
        ids_offset, spec = _scratch_layout(q, self.device)
        storage = scratch_tensor(scratch, (spec,), owner="short convolution")
        gathered, _ = materialize_scratch_view(
            storage,
            offset_bytes=0,
            shape=(q.max_seqs, q.channels, q.state_length),
            dtype=torch.bfloat16,
        )
        ids, _ = materialize_scratch_view(
            storage,
            offset_bytes=ids_offset,
            shape=(q.max_tokens,),
            dtype=torch.int32,
        )
        if q.mode == "mixed":
            if request_is_prefill is None:
                raise ValueError("mixed convolution requires request_is_prefill")
        elif request_is_prefill is not None:
            raise ValueError("request_is_prefill requires a mixed convolution plan")
        else:
            request_is_prefill = state_is_fresh
        tensors = dict(
            x=x,
            conv_weight=conv_weight,
            query_start_loc=query_start_loc,
            state_slot_ids=state_slot_ids,
            state_is_fresh=state_is_fresh,
            num_accepted_tokens=num_accepted_tokens,
            request_is_prefill=request_is_prefill,
            num_seqs=num_seqs,
            num_tokens=num_tokens,
            conv_state=conv_state,
            out=out,
            gathered_state=gathered,
            request_ids=ids,
        )
        shapes = {
            "x": (q.max_tokens, q.channels),
            "out": (q.max_tokens, q.channels),
            "conv_weight": (q.channels, q.kernel_size),
            "query_start_loc": (q.max_seqs + 1,),
            "num_seqs": (1,),
            "num_tokens": (1,),
        }
        for name in OPERANDS:
            if name in _SCRATCH or name in ("conv_state", "x"):
                continue
            dtype = (
                torch.bool
                if name in _BOOL
                else torch.int32
                if name in _I32
                else torch.int64
                if name == "state_slot_ids"
                else torch.bfloat16
            )
            _require_tensor(
                name,
                tensors[name],
                shape=shapes.get(name, (q.max_seqs,)),
                dtype=dtype,
                device=self.device,
            )
        if (
            x.shape != shapes["x"]
            or x.dtype != torch.bfloat16
            or x.device != self.device
        ):
            raise ValueError(
                "short convolution input must match declared BF16 shape and device"
            )
        _require_conv_state_tensor(
            conv_state,
            shape=(q.max_state_slots, q.channels, q.state_capacity),
            dtype=torch.bfloat16,
            device=self.device,
        )
        _require_mutation_alias_contract(
            mutable=(("scratch", storage), ("conv_state", conv_state), ("out", out)),
            read_only=tuple(
                (name, tensors[name])
                for name in OPERANDS
                if name not in _SCRATCH and name not in ("conv_state", "out")
            ),
        )
        self._check(tuple(tensors[name] for name in OPERANDS))
        return Binding(plan=plan, scratch=storage, **tensors)

    def run(self, binding):
        self.run_tensors(*(getattr(binding, name) for name in OPERANDS))
        return binding.out

    def run_tensors(
        self,
        x,
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
        gathered_state,
        request_ids,
    ):
        self._check(
            (
                x,
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
                gathered_state,
                request_ids,
            )
        )
        q, p = self.query, self.programs
        channels = triton.cdiv(q.channels, 128)
        with torch.cuda.device(self.device):
            if q.mode == "decode" and q.max_speculative_tokens > 0:
                p[0][(q.max_seqs + 1, channels, 1)](
                    x,
                    conv_weight,
                    query_start_loc,
                    state_slot_ids,
                    state_is_fresh,
                    num_accepted_tokens,
                    num_seqs,
                    num_tokens,
                    conv_state,
                    out,
                    *_speculative_args(q),
                )
                return
            if q.mode == "decode" and q.max_speculative_tokens == 0:
                p[0][(q.max_tokens, channels, 1)](
                    x,
                    conv_weight,
                    query_start_loc,
                    state_slot_ids,
                    state_is_fresh,
                    num_seqs,
                    num_tokens,
                    conv_state,
                    out,
                    q.channels,
                    q.kernel_size,
                    q.dilation,
                    q.input_row_stride,
                    q.state_strides[0],
                    128,
                )
                return
            p[0][(q.max_tokens, 1, 1)](
                query_start_loc,
                num_seqs,
                num_tokens,
                request_ids,
                q.max_tokens,
            )
            p[1][(q.max_seqs, channels, 1)](
                conv_state,
                state_slot_ids,
                state_is_fresh,
                num_accepted_tokens,
                request_is_prefill,
                num_seqs,
                gathered_state,
                *_state_args(q),
            )
            token_block = _token_block(q)
            if token_block > 1:
                p[2][(triton.cdiv(q.max_tokens, token_block), channels, 1)](
                    x,
                    gathered_state,
                    conv_weight,
                    query_start_loc,
                    request_ids,
                    state_slot_ids,
                    num_tokens,
                    out,
                    q.channels,
                    q.state_length,
                    q.kernel_size,
                    q.dilation,
                    128,
                    q.input_row_stride,
                    token_block,
                    q.max_tokens,
                )
            else:
                p[2][(q.max_tokens, channels, 1)](
                    x,
                    gathered_state,
                    conv_weight,
                    query_start_loc,
                    request_ids,
                    state_slot_ids,
                    num_tokens,
                    out,
                    q.channels,
                    q.state_length,
                    q.kernel_size,
                    q.dilation,
                    128,
                    False,
                    q.input_row_stride,
                )
            p[3][(q.max_seqs, channels, 1)](
                x,
                gathered_state,
                query_start_loc,
                state_slot_ids,
                request_is_prefill,
                num_seqs,
                conv_state,
                *_state_args(q),
                q.input_row_stride,
            )


def make_plan(caps, *, invocation=None, override=None):
    from .api import Caps

    if not isinstance(caps, Caps):
        raise TypeError("caps must be short_conv.Caps")
    invocation = FrozenMapping(invocation or {})
    if set(invocation) - {"state_strides", "input_alignments", "input_row_stride"}:
        raise ValueError("unknown short convolution invocation metadata")
    query = ShortConvQuery(
        **{
            name: getattr(caps, name)
            for name in Caps.__dataclass_fields__
            if name != "device"
        },
        **dict(invocation),
    )
    device = torch.device(caps.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())

    def jobs(config, detected):
        return (
            CompileJob.create(
                "b12x.sequence.short_conv._preparation:compile_pipeline",
                TUNING.encode_query(query),
                detected.ordinal,
            ),
        )

    def memory(config, detected):
        _, spec = _scratch_layout(query, torch.device("cuda", detected.ordinal))
        return MemoryRequirements(scratch=(spec,))

    def materialize(selection, detected):
        return PreparedShortConv(
            query,
            torch.device("cuda", detected.ordinal),
            compile_pipeline(TUNING.encode_query(query), detected.ordinal),
        )

    return Plan(
        contract=TUNING,
        query=query,
        invocation=invocation,
        override=override,
        shared=False,
        _device=device,
        _compile_jobs=jobs,
        _memory_requirements=memory,
        _materialize=materialize,
    )
