"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Paired frozen forward/backward training API for recurrent KDA."""

import math
import threading
from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence

import torch

from .api_logging import flashinfer_api
from .kda_backward import KDA_BACKWARD_GRADIENT_NAMES
from .utils import get_compute_capability

_HEAD_DIM = 128
_TOKENS = 8192
_SEQUENCES = 8
_HEADS = 96
_CHUNK = 16
_CHUNKS = _TOKENS // _CHUNK
_WORK_ITEMS = _SEQUENCES * _HEADS
_LOWER_BOUND = -5.0
_SCALE = 1.0 / math.sqrt(_HEAD_DIM)
_SUPPORTED_COMPUTE_CAPABILITIES = ((10, 0), (10, 3))
_CU_SEQLENS = tuple(range(0, _TOKENS + 1, _TOKENS // _SEQUENCES))
_TrainingTarget = Literal["sm100a", "sm103a"]


def _tensor_signature(tensor: torch.Tensor) -> tuple:
    return (
        tensor.data_ptr(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        int(tensor._version),
    )


def _storage_ranges_overlap(left: torch.Tensor, right: torch.Tensor) -> bool:
    if left.device != right.device or left.numel() == 0 or right.numel() == 0:
        return False

    def storage_end(tensor: torch.Tensor) -> int:
        max_element_offset = sum(
            (size - 1) * stride
            for size, stride in zip(tensor.shape, tensor.stride(), strict=True)
            if size > 0
        )
        return tensor.data_ptr() + (max_element_offset + 1) * tensor.element_size()

    left_start = left.data_ptr()
    right_start = right.data_ptr()
    return left_start < storage_end(right) and right_start < storage_end(left)


def _check_writes_do_not_overlap(
    writes: Sequence[tuple[str, torch.Tensor]],
    reads: Sequence[tuple[str, torch.Tensor]],
) -> None:
    for write_index, (write_name, write_tensor) in enumerate(writes):
        for read_name, read_tensor in reads:
            if _storage_ranges_overlap(write_tensor, read_tensor):
                raise ValueError(f"{write_name} must not overlap {read_name}")
        for other_name, other_tensor in writes[write_index + 1 :]:
            if _storage_ranges_overlap(write_tensor, other_tensor):
                raise ValueError(f"{write_name} must not overlap {other_name}")


def _validate_tensor(
    tensor: torch.Tensor,
    name: str,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_cuda or tensor.device != device:
        raise ValueError(f"{name} must be on CUDA device {device}")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _validate_scale_and_bound(
    scale: Optional[float], lower_bound: float
) -> tuple[float, float]:
    scale_value = _SCALE if scale is None else float(scale)
    lower_bound_value = float(lower_bound)
    if not math.isfinite(scale_value) or abs(scale_value - _SCALE) > 1e-15:
        raise ValueError(
            f"recurrent KDA training fixes scale=1/sqrt(128), got {scale_value}"
        )
    if not math.isfinite(lower_bound_value) or lower_bound_value != _LOWER_BOUND:
        raise ValueError(
            f"recurrent KDA training fixes lower_bound=-5.0, got {lower_bound_value}"
        )
    return scale_value, lower_bound_value


def _validate_forward_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> None:
    if not isinstance(q, torch.Tensor) or not q.is_cuda:
        raise ValueError("recurrent_kda_training_forward requires CUDA tensors")
    device = q.device
    if get_compute_capability(device) not in _SUPPORTED_COMPUTE_CAPABILITIES:
        raise ValueError(
            "recurrent_kda_training_forward requires compute capability 10.0 or 10.3"
        )
    token_shape = (1, _TOKENS, _HEADS, _HEAD_DIM)
    for name, tensor in (("q", q), ("k", k), ("v", v), ("g", g)):
        _validate_tensor(
            tensor, name, shape=token_shape, dtype=torch.bfloat16, device=device
        )
    _validate_tensor(
        beta,
        "beta",
        shape=token_shape[:-1],
        dtype=torch.bfloat16,
        device=device,
    )
    _validate_tensor(
        A_log, "A_log", shape=(_HEADS,), dtype=torch.float32, device=device
    )
    _validate_tensor(
        dt_bias,
        "dt_bias",
        shape=(_HEADS, _HEAD_DIM),
        dtype=torch.float32,
        device=device,
    )
    _validate_tensor(
        initial_state,
        "initial_state",
        shape=(_SEQUENCES, _HEADS, _HEAD_DIM, _HEAD_DIM),
        dtype=torch.float32,
        device=device,
    )
    _validate_tensor(
        cu_seqlens,
        "cu_seqlens",
        shape=(_SEQUENCES + 1,),
        dtype=torch.int64,
        device=device,
    )
    offsets = tuple(int(value) for value in cu_seqlens.detach().cpu().tolist())
    if offsets != _CU_SEQLENS:
        raise ValueError(
            "recurrent KDA training requires eight uniform 1024-token sequences"
        )


def _aligned_u8(device: torch.device, size: int) -> torch.Tensor:
    raw = torch.empty(size + 63, dtype=torch.uint8, device=device)
    offset = (-raw.data_ptr()) % 64
    return raw[offset : offset + size]


def _forward_work_items(device: torch.device) -> torch.Tensor:
    rows = [
        (
            sequence,
            head,
            0,
            64,
            0,
            64,
            sequence * 1024,
            (sequence + 1) * 1024,
        )
        for sequence in range(_SEQUENCES)
        for head in range(_HEADS)
    ]
    return torch.tensor(rows, dtype=torch.int32, device=device)


def _backward_work_items(device: torch.device) -> torch.Tensor:
    rows = [
        (sequence, head, 0, 64, 64)
        for sequence in range(_SEQUENCES)
        for head in range(_HEADS)
    ]
    return torch.tensor(rows, dtype=torch.int32, device=device)


def _checkpoint_cu_starts(device: torch.device) -> torch.Tensor:
    return torch.arange(
        0, _CHUNKS + 1, _CHUNKS // _SEQUENCES, dtype=torch.int64, device=device
    )


def _training_target(device: torch.device) -> _TrainingTarget:
    return "sm100a" if get_compute_capability(device) == (10, 0) else "sm103a"


def _get_training_module(device: torch.device):
    from .jit.flash_kda_training import load_flash_kda_training_module

    return load_flash_kda_training_module(_training_target(device))


def _get_backward_module(device: torch.device):
    from .jit.flash_kda_backward import load_flash_kda_backward_module

    return load_flash_kda_backward_module(_training_target(device))


@dataclass
class RecurrentKDATrainingContext:
    """Persistent forward state consumed by :func:`recurrent_kda_training_backward`.

    The public checkpoint and active-beta tensors are the exact values saved by
    the frozen forward kernel. The remaining fields retain input and launch
    storage lifetimes and are intentionally excluded from the representation.
    """

    state_checkpoints: torch.Tensor
    beta_active: torch.Tensor
    _q: torch.Tensor = field(repr=False)
    _k: torch.Tensor = field(repr=False)
    _v: torch.Tensor = field(repr=False)
    _g: torch.Tensor = field(repr=False)
    _beta: torch.Tensor = field(repr=False)
    _A_log: torch.Tensor = field(repr=False)
    _dt_bias: torch.Tensor = field(repr=False)
    _initial_state: torch.Tensor = field(repr=False)
    _cu_seqlens: torch.Tensor = field(repr=False)
    _final_state_bf16: torch.Tensor = field(repr=False)
    _final_state_recurrence_output: torch.Tensor = field(repr=False)
    _forward_work_items: torch.Tensor = field(repr=False)
    _backward_work_items: torch.Tensor = field(repr=False)
    _checkpoint_cu_starts: torch.Tensor = field(repr=False)
    _forward_descriptor_storage: torch.Tensor = field(repr=False)
    _input_signatures: tuple[tuple, ...] = field(repr=False)
    _saved_context_signatures: tuple[tuple, ...] = field(repr=False)
    _stream_ptr: int = field(repr=False)
    _forward_descriptor_signature: Optional[tuple] = field(default=None, repr=False)
    _backward_descriptor_storage: Optional[torch.Tensor] = field(
        default=None, repr=False
    )
    _backward_descriptor_signature: Optional[tuple] = field(default=None, repr=False)
    _backward_buffers: dict[str, torch.Tensor] = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


def _validate_descriptor_storage(
    tensor: torch.Tensor,
    name: str,
    *,
    device: torch.device,
    minimum_bytes: int,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_cuda or tensor.device != device:
        raise ValueError(f"{name} must be on CUDA device {device}")
    if tensor.dtype != torch.uint8 or not tensor.is_contiguous():
        raise ValueError(f"{name} must be a contiguous uint8 tensor")
    if tensor.numel() < minimum_bytes:
        raise ValueError(f"{name} must provide at least {minimum_bytes} bytes")
    if tensor.data_ptr() % 64 != 0:
        raise ValueError(f"{name} must be 64-byte aligned")


def _validate_context_storage(
    context: RecurrentKDATrainingContext, device: torch.device
) -> None:
    for name, tensor, shape, dtype in (
        (
            "context.state_checkpoints",
            context.state_checkpoints,
            (_CHUNKS, _HEADS, _HEAD_DIM, _HEAD_DIM),
            torch.bfloat16,
        ),
        (
            "context.beta_active",
            context.beta_active,
            (_TOKENS, _HEADS),
            torch.bfloat16,
        ),
        (
            "context._final_state_bf16",
            context._final_state_bf16,
            (_SEQUENCES, _HEADS, _HEAD_DIM, _HEAD_DIM),
            torch.bfloat16,
        ),
        (
            "context._final_state_recurrence_output",
            context._final_state_recurrence_output,
            (1, _TOKENS, _HEADS, _HEAD_DIM),
            torch.bfloat16,
        ),
        (
            "context._forward_work_items",
            context._forward_work_items,
            (_WORK_ITEMS, 8),
            torch.int32,
        ),
        (
            "context._backward_work_items",
            context._backward_work_items,
            (_WORK_ITEMS, 5),
            torch.int32,
        ),
        (
            "context._checkpoint_cu_starts",
            context._checkpoint_cu_starts,
            (_SEQUENCES + 1,),
            torch.int64,
        ),
    ):
        _validate_tensor(tensor, name, shape=shape, dtype=dtype, device=device)
    _validate_descriptor_storage(
        context._forward_descriptor_storage,
        "context._forward_descriptor_storage",
        device=device,
        minimum_bytes=6 * 128,
    )
    if context._backward_descriptor_storage is not None:
        _validate_descriptor_storage(
            context._backward_descriptor_storage,
            "context._backward_descriptor_storage",
            device=device,
            minimum_bytes=8 * 128,
        )


def _saved_context_tensors(
    context: RecurrentKDATrainingContext,
) -> tuple[torch.Tensor, ...]:
    return (
        context.state_checkpoints,
        context.beta_active,
        context._backward_work_items,
    )


def _run_final_state_recurrence(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    lower_bound: float,
    output_scratch: torch.Tensor,
    state_scratch: torch.Tensor,
    final_state: torch.Tensor,
) -> None:
    from . import kda_prefill as _kda_prefill

    state_scratch.copy_(initial_state)
    _kda_prefill._run_flash_kda_prefill(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=state_scratch,
        output_final_state=True,
        lower_bound=lower_bound,
        cu_seqlens=cu_seqlens,
        output=output_scratch,
        seq_order=None,
        prefill_workspace=None,
        state_indices=None,
        state_checkpoints=None,
        checkpoint_cu_starts=None,
        checkpoint_every_n_tokens=0,
    )
    final_state.copy_(state_scratch)


def _recurrent_kda_training_forward_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: Optional[float] = None,
    lower_bound: float = _LOWER_BOUND,
    out: Optional[torch.Tensor] = None,
    final_state_out: Optional[torch.Tensor] = None,
    context_out: Optional[RecurrentKDATrainingContext] = None,
) -> tuple[torch.Tensor, torch.Tensor, RecurrentKDATrainingContext]:
    r"""Run the checkpoint-producing recurrent KDA training forward.

    This exact Blackwell route accepts eight packed 1024-token sequences and
    96 heads. It returns BF16 token output, a BF16 serving-recurrence final state
    promoted to FP32, and a persistent context. The paired backward consumes the
    checkpoint-producing context directly and never reruns the forward
    recurrence.
    """

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("recurrent KDA training does not support CUDA graph capture")
    _validate_forward_inputs(
        q, k, v, g, beta, A_log, dt_bias, initial_state, cu_seqlens
    )
    scale_value, lower_bound_value = _validate_scale_and_bound(scale, lower_bound)
    device = q.device
    stream_ptr = int(torch.cuda.current_stream(device).cuda_stream)
    output = torch.empty_like(q) if out is None else out
    final_state = (
        torch.empty_like(initial_state) if final_state_out is None else final_state_out
    )
    _validate_tensor(
        output,
        "out",
        shape=tuple(q.shape),
        dtype=torch.bfloat16,
        device=device,
    )
    _validate_tensor(
        final_state,
        "final_state_out",
        shape=tuple(initial_state.shape),
        dtype=torch.float32,
        device=device,
    )
    forward_inputs = (
        ("q", q),
        ("k", k),
        ("v", v),
        ("g", g),
        ("beta", beta),
        ("A_log", A_log),
        ("dt_bias", dt_bias),
        ("initial_state", initial_state),
        ("cu_seqlens", cu_seqlens),
    )
    _check_writes_do_not_overlap(
        (("out", output), ("final_state_out", final_state)), forward_inputs
    )

    if context_out is None:
        final_state_bf16 = torch.empty_like(initial_state, dtype=torch.bfloat16)
        final_state_recurrence_output = torch.empty_like(q)
        state_checkpoints = torch.empty(
            (_CHUNKS, _HEADS, _HEAD_DIM, _HEAD_DIM),
            dtype=torch.bfloat16,
            device=device,
        )
        beta_active = torch.empty(
            (_TOKENS, _HEADS), dtype=torch.bfloat16, device=device
        )
        forward_items = _forward_work_items(device)
        backward_items = _backward_work_items(device)
        checkpoint_starts = _checkpoint_cu_starts(device)
        descriptor_storage = _aligned_u8(device, 6 * 128)
        prepare_descriptors = 1
    else:
        if not isinstance(context_out, RecurrentKDATrainingContext):
            raise TypeError("context_out must be a RecurrentKDATrainingContext")
        if context_out._stream_ptr != stream_ptr:
            raise RuntimeError(
                "a recurrent KDA training context must be reused on its forward stream"
            )
        _validate_context_storage(context_out, device)
        final_state_bf16 = context_out._final_state_bf16
        final_state_recurrence_output = context_out._final_state_recurrence_output
        state_checkpoints = context_out.state_checkpoints
        beta_active = context_out.beta_active
        forward_items = context_out._forward_work_items
        backward_items = context_out._backward_work_items
        checkpoint_starts = context_out._checkpoint_cu_starts
        descriptor_storage = context_out._forward_descriptor_storage
        prepare_descriptors = int(
            context_out._forward_descriptor_signature
            != tuple(
                _tensor_signature(tensor)
                for tensor in (q, k, v, g, output, state_checkpoints)
            )
        )
    _check_writes_do_not_overlap(
        (
            ("out", output),
            ("final_state_out", final_state),
            ("context._final_state_bf16", final_state_bf16),
            (
                "context._final_state_recurrence_output",
                final_state_recurrence_output,
            ),
            ("context.state_checkpoints", state_checkpoints),
            ("context.beta_active", beta_active),
            ("context._forward_descriptor_storage", descriptor_storage),
        ),
        (
            *forward_inputs,
            ("context._forward_work_items", forward_items),
            ("context._backward_work_items", backward_items),
            ("context._checkpoint_cu_starts", checkpoint_starts),
        ),
    )
    counter = torch.zeros(1, dtype=torch.uint32, device=device)
    _get_training_module(device).run_forward(
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        initial_state,
        cu_seqlens,
        checkpoint_starts,
        forward_items,
        descriptor_storage,
        output,
        final_state_bf16,
        final_state,
        state_checkpoints,
        beta_active,
        counter,
        prepare_descriptors,
        _SEQUENCES,
        _HEADS,
        scale_value,
        lower_bound_value,
        stream_ptr,
    )
    _run_final_state_recurrence(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        scale=scale_value,
        lower_bound=lower_bound_value,
        output_scratch=final_state_recurrence_output,
        state_scratch=final_state_bf16,
        final_state=final_state,
    )
    inputs = (q, k, v, g, beta, A_log, dt_bias, initial_state, cu_seqlens)
    descriptor_signature = tuple(
        _tensor_signature(tensor) for tensor in (q, k, v, g, output, state_checkpoints)
    )
    if context_out is None:
        context = RecurrentKDATrainingContext(
            state_checkpoints=state_checkpoints,
            beta_active=beta_active,
            _q=q,
            _k=k,
            _v=v,
            _g=g,
            _beta=beta,
            _A_log=A_log,
            _dt_bias=dt_bias,
            _initial_state=initial_state,
            _cu_seqlens=cu_seqlens,
            _final_state_bf16=final_state_bf16,
            _final_state_recurrence_output=final_state_recurrence_output,
            _forward_work_items=forward_items,
            _backward_work_items=backward_items,
            _checkpoint_cu_starts=checkpoint_starts,
            _forward_descriptor_storage=descriptor_storage,
            _input_signatures=tuple(_tensor_signature(tensor) for tensor in inputs),
            _saved_context_signatures=(),
            _stream_ptr=stream_ptr,
            _forward_descriptor_signature=descriptor_signature,
        )
    else:
        context = context_out
        context._q = q
        context._k = k
        context._v = v
        context._g = g
        context._beta = beta
        context._A_log = A_log
        context._dt_bias = dt_bias
        context._initial_state = initial_state
        context._cu_seqlens = cu_seqlens
        context._input_signatures = tuple(
            _tensor_signature(tensor) for tensor in inputs
        )
        context._forward_descriptor_signature = descriptor_signature
    context._saved_context_signatures = tuple(
        _tensor_signature(tensor) for tensor in _saved_context_tensors(context)
    )
    return output, final_state, context


@flashinfer_api
def recurrent_kda_training_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: Optional[float] = None,
    lower_bound: float = _LOWER_BOUND,
    out: Optional[torch.Tensor] = None,
    final_state_out: Optional[torch.Tensor] = None,
    context_out: Optional[RecurrentKDATrainingContext] = None,
) -> tuple[torch.Tensor, torch.Tensor, RecurrentKDATrainingContext]:
    r"""Run the checkpoint-producing recurrent KDA training forward.

    This exact Blackwell route accepts eight packed 1024-token sequences and
    96 heads. It returns BF16 token output, a BF16 serving-recurrence final state
    promoted to FP32, and a persistent checkpoint-producing context. The paired
    backward consumes that context directly and never reruns the forward
    recurrence. Backward and any ``context_out`` reuse must run on the CUDA
    stream that produced the context; calls sharing a context are serialized.
    """

    if context_out is None:
        return _recurrent_kda_training_forward_impl(
            q,
            k,
            v,
            g,
            beta,
            A_log,
            dt_bias,
            initial_state,
            cu_seqlens,
            scale,
            lower_bound,
            out,
            final_state_out,
            None,
        )
    if not isinstance(context_out, RecurrentKDATrainingContext):
        raise TypeError("context_out must be a RecurrentKDATrainingContext")
    with context_out._lock:
        return _recurrent_kda_training_forward_impl(
            q,
            k,
            v,
            g,
            beta,
            A_log,
            dt_bias,
            initial_state,
            cu_seqlens,
            scale,
            lower_bound,
            out,
            final_state_out,
            context_out,
        )


def _validate_gradient_outputs(
    out: Optional[Sequence[torch.Tensor]], context: RecurrentKDATrainingContext
) -> tuple[torch.Tensor, ...]:
    expected = (
        (context._q.shape, torch.bfloat16),
        (context._q.shape, torch.bfloat16),
        (context._q.shape, torch.bfloat16),
        (context._q.shape, torch.bfloat16),
        ((1, _TOKENS, _HEADS), torch.bfloat16),
        ((_HEADS,), torch.float32),
        ((_HEADS, _HEAD_DIM), torch.float32),
        ((_SEQUENCES, _HEADS, _HEAD_DIM, _HEAD_DIM), torch.float32),
    )
    if out is None:
        return tuple(
            torch.empty(shape, dtype=dtype, device=context._q.device)
            for shape, dtype in expected
        )
    if len(out) != len(expected):
        raise ValueError("out must contain eight gradient tensors")
    for name, tensor, (shape, dtype) in zip(
        KDA_BACKWARD_GRADIENT_NAMES, out, expected, strict=True
    ):
        _validate_tensor(
            tensor, name, shape=tuple(shape), dtype=dtype, device=context._q.device
        )
    return tuple(out)


@flashinfer_api
def recurrent_kda_training_backward(
    context: RecurrentKDATrainingContext,
    do: torch.Tensor,
    dfinal_state: torch.Tensor,
    out: Optional[Sequence[torch.Tensor]] = None,
) -> tuple[torch.Tensor, ...]:
    r"""Differentiate a saved context on the CUDA stream that produced it."""

    if not isinstance(context, RecurrentKDATrainingContext):
        raise TypeError("context must be a RecurrentKDATrainingContext")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("recurrent KDA training does not support CUDA graph capture")
    with context._lock:
        device = context._q.device
        stream_ptr = int(torch.cuda.current_stream(device).cuda_stream)
        if stream_ptr != context._stream_ptr:
            raise RuntimeError(
                "recurrent KDA training backward must run on the forward stream"
            )
        _validate_context_storage(context, device)
        inputs = (
            context._q,
            context._k,
            context._v,
            context._g,
            context._beta,
            context._A_log,
            context._dt_bias,
            context._initial_state,
            context._cu_seqlens,
        )
        observed = tuple(_tensor_signature(tensor) for tensor in inputs)
        if observed != context._input_signatures:
            raise RuntimeError(
                "a recurrent KDA training input was modified after forward"
            )
        saved_context_signatures = tuple(
            _tensor_signature(tensor) for tensor in _saved_context_tensors(context)
        )
        if saved_context_signatures != context._saved_context_signatures:
            raise RuntimeError(
                "the recurrent KDA training context was modified after forward"
            )
        _validate_tensor(
            do,
            "do",
            shape=tuple(context._q.shape),
            dtype=torch.bfloat16,
            device=device,
        )
        _validate_tensor(
            dfinal_state,
            "dfinal_state",
            shape=(_SEQUENCES, _HEADS, _HEAD_DIM, _HEAD_DIM),
            dtype=torch.float32,
            device=device,
        )
        outputs = _validate_gradient_outputs(out, context)
        dq, dk, dv, dg, dbeta, dA_log, ddt_bias, dinitial_state = outputs

        def saved_buffer(
            name: str, shape: tuple[int, ...], dtype: torch.dtype
        ) -> torch.Tensor:
            tensor = context._backward_buffers.get(name)
            if tensor is None:
                tensor = torch.empty(shape, dtype=dtype, device=device)
                context._backward_buffers[name] = tensor
            else:
                _validate_tensor(
                    tensor,
                    f"context._backward_buffers[{name!r}]",
                    shape=shape,
                    dtype=dtype,
                    device=device,
                )
            return tensor

        dlog_decay = saved_buffer(
            "dlog_decay", (_TOKENS, _HEADS, _HEAD_DIM), torch.float32
        )
        dlog_boundary = saved_buffer(
            "dlog_boundary", (_CHUNKS, _HEADS, _HEAD_DIM), torch.float32
        )
        dbeta_active = saved_buffer("dbeta_active", (_TOKENS, _HEADS), torch.float32)
        gate_part_a = saved_buffer(
            "gate_part_a", (128, _HEADS, _HEAD_DIM), torch.float32
        )
        gate_part_dt = saved_buffer(
            "gate_part_dt", (128, _HEADS, _HEAD_DIM), torch.float32
        )
        counter = saved_buffer("counter", (1,), torch.uint32)
        dummy_u32 = saved_buffer("dummy_u32", (1,), torch.uint32)
        dummy_f32 = saved_buffer("dummy_f32", (1,), torch.float32)
        if context._backward_descriptor_storage is None:
            context._backward_descriptor_storage = _aligned_u8(device, 8 * 128)
        descriptor_storage = context._backward_descriptor_storage
        _validate_descriptor_storage(
            descriptor_storage,
            "context._backward_descriptor_storage",
            device=device,
            minimum_bytes=8 * 128,
        )
        backward_writes = (
            ("dq", dq),
            ("dk", dk),
            ("dv", dv),
            ("dg", dg),
            ("dbeta", dbeta),
            ("dA_log", dA_log),
            ("ddt_bias", ddt_bias),
            ("dinitial_state", dinitial_state),
            ("context.dlog_decay", dlog_decay),
            ("context.dlog_boundary", dlog_boundary),
            ("context.dbeta_active", dbeta_active),
            ("context.gate_part_a", gate_part_a),
            ("context.gate_part_dt", gate_part_dt),
            ("context.counter", counter),
            ("context.dummy_u32", dummy_u32),
            ("context.dummy_f32", dummy_f32),
            ("context._backward_descriptor_storage", descriptor_storage),
        )
        _check_writes_do_not_overlap(
            backward_writes,
            (
                *(
                    zip(
                        (
                            "q",
                            "k",
                            "v",
                            "g",
                            "beta",
                            "A_log",
                            "dt_bias",
                            "initial_state",
                            "cu_seqlens",
                        ),
                        inputs,
                        strict=True,
                    )
                ),
                ("do", do),
                ("dfinal_state", dfinal_state),
                ("context.state_checkpoints", context.state_checkpoints),
                ("context.beta_active", context.beta_active),
                ("context._backward_work_items", context._backward_work_items),
            ),
        )
        descriptor_signature = tuple(
            _tensor_signature(tensor)
            for tensor in (
                context._q,
                context._k,
                context._v,
                context._g,
                do,
                context.state_checkpoints,
                dv,
                context.beta_active,
            )
        )
        prepare_descriptors = int(
            descriptor_signature != context._backward_descriptor_signature
        )
        _get_backward_module(device).run_c16_backward(
            context._q,
            context._k,
            context._v,
            context._g,
            context._A_log,
            context._dt_bias,
            do,
            dfinal_state,
            context._cu_seqlens,
            context._backward_work_items,
            descriptor_storage,
            context.state_checkpoints,
            context.beta_active,
            dlog_decay,
            dlog_boundary,
            dbeta_active,
            gate_part_a,
            gate_part_dt,
            counter,
            dummy_u32,
            dummy_f32,
            dq,
            dk,
            dv,
            dg,
            dbeta,
            dA_log,
            ddt_bias,
            dinitial_state,
            prepare_descriptors,
            _SEQUENCES,
            _HEADS,
            _SCALE,
            _LOWER_BOUND,
            stream_ptr,
        )
        context._backward_descriptor_signature = descriptor_signature
        return outputs


__all__ = [
    "RecurrentKDATrainingContext",
    "recurrent_kda_training_backward",
    "recurrent_kda_training_forward",
]
