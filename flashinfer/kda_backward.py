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

# Frozen Blackwell recurrent Kimi Delta Attention backward support.

import math
import threading
from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from .api_logging import flashinfer_api
from .utils import get_compute_capability


_HEAD_DIM = 128
_CHUNK_SIZE = 32
_C16_CHUNK_SIZE = 16
_DESCRIPTOR_BYTES = 1152
_DESCRIPTOR_ALIGNMENT = 64
_SUPPORTED_COMPUTE_CAPABILITIES = ((10, 0), (10, 3))
_DEFAULT_LOWER_BOUND = -5.0

KDA_BACKWARD_GRADIENT_NAMES = (
    "dq",
    "dk",
    "dv",
    "dg",
    "dbeta",
    "dA_log",
    "ddt_bias",
    "dinitial_state",
)


@dataclass(frozen=True)
class _ShapeSpec:
    name: str
    seq_lens: tuple[int, ...]
    num_heads: int
    packed: bool

    @property
    def total_tokens(self) -> int:
        return sum(self.seq_lens)

    @property
    def num_sequences(self) -> int:
        return len(self.seq_lens)

    @property
    def high_head_route(self) -> bool:
        return self.num_heads >= 16

    @property
    def c16_route(self) -> bool:
        # The exported V483 binding intentionally freezes the one measured
        # production crossover shape. Other admitted shapes retain their
        # established low-head or C32 route.
        return self.packed and self.seq_lens == (1024,) * 8 and self.num_heads == 96

    @property
    def cu_seqlens(self) -> tuple[int, ...]:
        offsets = [0]
        for length in self.seq_lens:
            offsets.append(offsets[-1] + length)
        return tuple(offsets)


_SUPPORTED_SHAPES = (
    _ShapeSpec("fixed_t17_h1", (17,), 1, False),
    _ShapeSpec("packed_17_33_65_h4", (17, 33, 65), 4, True),
    _ShapeSpec("fixed_t17_h16", (17,), 16, False),
    _ShapeSpec("fixed_t1024_h4", (1024,), 4, False),
    _ShapeSpec("fixed_t4096_h32", (4096,), 32, False),
    _ShapeSpec("fixed_t8192_h96", (8192,), 96, False),
    _ShapeSpec(
        "packed_1300_547_2048_963_271_3063_h96",
        (1300, 547, 2048, 963, 271, 3063),
        96,
        True,
    ),
    _ShapeSpec("packed_1024x8_h96", (1024,) * 8, 96, True),
)


def _select_shape(
    q_shape: Sequence[int], cu_seqlens_numel: Optional[int]
) -> _ShapeSpec:
    if len(q_shape) != 4:
        raise ValueError(f"q must have rank 4, got shape {tuple(q_shape)}")
    batch_size, total_tokens, num_heads, head_dim = map(int, q_shape)
    packed = cu_seqlens_numel is not None
    num_sequences = None if cu_seqlens_numel is None else cu_seqlens_numel - 1
    for spec in _SUPPORTED_SHAPES:
        if (
            batch_size == 1
            and total_tokens == spec.total_tokens
            and num_heads == spec.num_heads
            and head_dim == _HEAD_DIM
            and packed == spec.packed
            and (not packed or num_sequences == spec.num_sequences)
        ):
            return spec
    layout = (
        "fixed"
        if cu_seqlens_numel is None
        else f"packed with {num_sequences} sequences"
    )
    raise ValueError(
        "recurrent_kda_backward supports only its eight documented Blackwell "
        f"shapes; got {layout} q shape {tuple(q_shape)}"
    )


def _tensor_signature(tensor: torch.Tensor) -> tuple:
    return (
        tensor.data_ptr(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
    )


def _metadata_values(spec: _ShapeSpec) -> dict[str, tuple[int, ...]]:
    chunk_counts = tuple(
        (length + _CHUNK_SIZE - 1) // _CHUNK_SIZE for length in spec.seq_lens
    )
    chunk_offsets = [0]
    chunk_sequence: list[int] = []
    chunk_index: list[int] = []
    for sequence, count in enumerate(chunk_counts):
        chunk_offsets.append(chunk_offsets[-1] + count)
        chunk_sequence.extend((sequence,) * count)
        chunk_index.extend(range(count))

    seq_order = tuple(
        sorted(
            range(spec.num_sequences),
            key=lambda sequence: spec.seq_lens[sequence],
            reverse=True,
        )
    )
    consumer_chunk_order = tuple(
        chunk_offsets[sequence] + chunk_counts[sequence] - 1 - reverse_depth
        for reverse_depth in range(max(chunk_counts))
        for sequence in seq_order
        if reverse_depth < chunk_counts[sequence]
    )
    chunk_pair_start = tuple(
        chunk_offsets[sequence] + local_chunk
        for sequence, count in enumerate(chunk_counts)
        for local_chunk in range(0, count, 2)
    )
    result = {
        "fixed_cu_seqlens": spec.cu_seqlens,
        "seq_order": seq_order,
        "cu_chunk_offsets": tuple(chunk_offsets),
        "consumer_chunk_order": consumer_chunk_order,
        "chunk_sequence": tuple(chunk_sequence),
        "chunk_index": tuple(chunk_index),
        "chunk_pair_start": chunk_pair_start,
    }
    if spec.c16_route:
        counts = tuple(length // _C16_CHUNK_SIZE for length in spec.seq_lens)
        checkpoint_starts = [0]
        for count in counts:
            checkpoint_starts.append(checkpoint_starts[-1] + count)
        forward_rows = tuple(
            item
            for sequence, count in enumerate(counts)
            for head in range(spec.num_heads)
            for item in (
                sequence,
                head,
                0,
                count,
                0,
                count,
                spec.cu_seqlens[sequence],
                spec.cu_seqlens[sequence + 1],
            )
        )
        backward_rows = tuple(
            item
            for sequence, count in enumerate(counts)
            for head in range(spec.num_heads)
            for item in (sequence, head, 0, count, count)
        )
        result.update(
            {
                "c16_checkpoint_cu_starts": tuple(checkpoint_starts),
                "c16_forward_work_items": forward_rows,
                "c16_backward_work_items": backward_rows,
            }
        )
    return result


class _RecurrentKDABackwardWorkspaceBase:
    def __init__(self, device: torch.device | str) -> None:
        normalized_device = torch.device(device)
        if normalized_device.type != "cuda":
            raise ValueError("RecurrentKDABackwardWorkspace requires a CUDA device")
        if normalized_device.index is None:
            normalized_device = torch.device("cuda", torch.cuda.current_device())
        self.device = normalized_device
        self._lock = threading.Lock()
        self._buffers: dict[str, torch.Tensor] = {}
        self._metadata: dict[str, torch.Tensor] = {}
        self._metadata_shape_name: Optional[str] = None
        self._descriptor_raw = torch.empty(
            _DESCRIPTOR_BYTES + _DESCRIPTOR_ALIGNMENT - 1,
            dtype=torch.uint8,
            device=self.device,
        )
        descriptor_offset = (-self._descriptor_raw.data_ptr()) % _DESCRIPTOR_ALIGNMENT
        self._descriptor_storage = self._descriptor_raw[
            descriptor_offset : descriptor_offset + _DESCRIPTOR_BYTES
        ]
        self._descriptor_signature: Optional[tuple] = None
        self._forward_descriptor_raw = torch.empty(
            6 * 128 + _DESCRIPTOR_ALIGNMENT - 1,
            dtype=torch.uint8,
            device=self.device,
        )
        forward_descriptor_offset = (
            -self._forward_descriptor_raw.data_ptr()
        ) % _DESCRIPTOR_ALIGNMENT
        self._forward_descriptor_storage = self._forward_descriptor_raw[
            forward_descriptor_offset : forward_descriptor_offset + 6 * 128
        ]
        self._forward_descriptor_signature: Optional[tuple] = None
        self._warmed_signature: Optional[tuple] = None
        self._packed_offsets_signature: Optional[tuple] = None
        self._bound_stream_ptr: Optional[int] = None
        self._captured = False


class RecurrentKDABackwardWorkspace(_RecurrentKDABackwardWorkspaceBase):
    """Caller-owned scratch for :func:`recurrent_kda_backward`.

    Construct one workspace per invocation that will be captured by a CUDA
    Graph. Warm it by calling :func:`recurrent_kda_backward` eagerly with the
    exact input and ``out`` tensors on the intended capture stream, then
    synchronize that stream before capture. The warm call allocates every
    route-specific intermediate, validates packed offsets, and prepares the
    high-head TMA descriptors. Capture performs no allocation or descriptor
    preparation.

    A workspace binds to its first stream. Once it participates in capture it
    cannot be passed through Python again; graph replay remains valid while
    the workspace and all warmed tensors stay alive.
    """


class _StreamWorkspace(_RecurrentKDABackwardWorkspaceBase):
    """Internal eager-only workspace for one device stream."""


_stream_workspaces: dict[tuple[int, int], _StreamWorkspace] = {}
_stream_workspaces_lock = threading.Lock()


def _get_stream_workspace(device: torch.device, stream_ptr: int) -> _StreamWorkspace:
    device_index = device.index
    assert device_index is not None
    key = (device_index, stream_ptr)
    with _stream_workspaces_lock:
        workspace = _stream_workspaces.get(key)
        if workspace is None:
            workspace = _StreamWorkspace(device)
            _stream_workspaces[key] = workspace
        return workspace


def _bind_workspace(
    workspace: _RecurrentKDABackwardWorkspaceBase,
    *,
    device: torch.device,
    stream_ptr: int,
    capturing: bool,
    explicit: bool,
) -> None:
    if workspace.device != device:
        raise ValueError(
            "RecurrentKDABackwardWorkspace is bound to "
            f"{workspace.device}, but inputs are on {device}"
        )
    if workspace._bound_stream_ptr is None:
        workspace._bound_stream_ptr = stream_ptr
    elif workspace._bound_stream_ptr != stream_ptr:
        raise RuntimeError(
            "RecurrentKDABackwardWorkspace is bound to a different CUDA "
            "stream; warm and capture it on one stream"
        )
    if explicit and workspace._captured:
        reuse_kind = "captured again" if capturing else "reused eagerly"
        raise RuntimeError(
            "RecurrentKDABackwardWorkspace has participated in CUDA graph "
            f"capture and cannot be {reuse_kind} or mutated"
        )


def _buffer(
    workspace: _RecurrentKDABackwardWorkspaceBase,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    *,
    capturing: bool,
) -> torch.Tensor:
    numel = math.prod(shape)
    allocation = workspace._buffers.get(name)
    if allocation is None or allocation.numel() < numel:
        if capturing:
            raise RuntimeError(
                "RecurrentKDABackwardWorkspace is not large enough for CUDA "
                "graph capture; eagerly warm this exact shape first"
            )
        allocation = torch.empty(numel, dtype=dtype, device=workspace.device)
        workspace._buffers[name] = allocation
    elif allocation.dtype != dtype:
        raise RuntimeError(
            f"workspace buffer {name} has dtype {allocation.dtype}, expected {dtype}"
        )
    return allocation[:numel].view(shape)


def _prepare_metadata(
    workspace: _RecurrentKDABackwardWorkspaceBase,
    spec: _ShapeSpec,
    *,
    capturing: bool,
) -> dict[str, torch.Tensor]:
    if workspace._metadata_shape_name == spec.name:
        return workspace._metadata
    if capturing:
        raise RuntimeError(
            "RecurrentKDABackwardWorkspace metadata was not warmed for "
            f"shape {spec.name}"
        )

    values = _metadata_values(spec)
    result: dict[str, torch.Tensor] = {}
    int64_metadata = {
        "fixed_cu_seqlens",
        "cu_chunk_offsets",
        "c16_checkpoint_cu_starts",
    }
    for name, items in values.items():
        dtype = torch.int64 if name in int64_metadata else torch.int32
        result[name] = torch.tensor(items, dtype=dtype, device=workspace.device)
    workspace._metadata = result
    workspace._metadata_shape_name = spec.name
    return result


def _validate_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.device != device or not tensor.is_cuda:
        raise ValueError(f"{name} must be on CUDA device {device}")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _validate_and_select_shape(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor,
    do: torch.Tensor,
    dfinal_state: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor],
) -> _ShapeSpec:
    if not isinstance(q, torch.Tensor):
        raise TypeError("q must be a torch.Tensor")
    if not q.is_cuda:
        raise ValueError("recurrent_kda_backward requires CUDA tensors")
    device = q.device
    compute_capability = get_compute_capability(device)
    if compute_capability not in _SUPPORTED_COMPUTE_CAPABILITIES:
        raise ValueError(
            "recurrent_kda_backward requires compute capability 10.0 or 10.3"
        )
    if cu_seqlens is not None and not isinstance(cu_seqlens, torch.Tensor):
        raise TypeError("cu_seqlens must be a torch.Tensor or None")
    spec = _select_shape(q.shape, None if cu_seqlens is None else cu_seqlens.numel())
    token_shape = (1, spec.total_tokens, spec.num_heads, _HEAD_DIM)
    beta_shape = token_shape[:-1]
    state_shape = (
        spec.num_sequences,
        spec.num_heads,
        _HEAD_DIM,
        _HEAD_DIM,
    )
    for name, tensor in (
        ("q", q),
        ("k", k),
        ("v", v),
        ("g", g),
        ("do", do),
    ):
        _validate_tensor(
            name, tensor, shape=token_shape, dtype=torch.bfloat16, device=device
        )
    _validate_tensor(
        "beta", beta, shape=beta_shape, dtype=torch.bfloat16, device=device
    )
    _validate_tensor(
        "A_log",
        A_log,
        shape=(spec.num_heads,),
        dtype=torch.float32,
        device=device,
    )
    _validate_tensor(
        "dt_bias",
        dt_bias,
        shape=(spec.num_heads, _HEAD_DIM),
        dtype=torch.float32,
        device=device,
    )
    _validate_tensor(
        "initial_state",
        initial_state,
        shape=state_shape,
        dtype=torch.float32,
        device=device,
    )
    _validate_tensor(
        "dfinal_state",
        dfinal_state,
        shape=state_shape,
        dtype=torch.float32,
        device=device,
    )
    if spec.packed:
        assert cu_seqlens is not None
        _validate_tensor(
            "cu_seqlens",
            cu_seqlens,
            shape=(spec.num_sequences + 1,),
            dtype=torch.int64,
            device=device,
        )
    elif cu_seqlens is not None:
        raise ValueError(f"shape {spec.name} requires cu_seqlens=None")
    return spec


def _validate_packed_offsets(
    workspace: _RecurrentKDABackwardWorkspaceBase,
    spec: _ShapeSpec,
    cu_seqlens: Optional[torch.Tensor],
    *,
    capturing: bool,
) -> None:
    if not spec.packed:
        return
    assert cu_seqlens is not None
    signature = (*_tensor_signature(cu_seqlens), int(cu_seqlens._version))
    if workspace._packed_offsets_signature == signature:
        return
    if capturing:
        raise RuntimeError(
            "packed cu_seqlens was not warmed for this CUDA graph capture"
        )
    offsets = tuple(int(value) for value in cu_seqlens.detach().cpu().tolist())
    if offsets != spec.cu_seqlens:
        raise ValueError(
            f"shape {spec.name} requires cu_seqlens={spec.cu_seqlens}, got {offsets}"
        )
    workspace._packed_offsets_signature = signature


def _validate_outputs(
    out: Optional[Sequence[torch.Tensor]],
    *,
    q: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor,
    capturing: bool,
) -> tuple[torch.Tensor, ...]:
    expected = (
        ("dq", q.shape, torch.bfloat16),
        ("dk", q.shape, torch.bfloat16),
        ("dv", q.shape, torch.bfloat16),
        ("dg", q.shape, torch.bfloat16),
        ("dbeta", beta.shape, torch.bfloat16),
        ("dA_log", A_log.shape, torch.float32),
        ("ddt_bias", dt_bias.shape, torch.float32),
        ("dinitial_state", initial_state.shape, torch.float32),
    )
    if out is None:
        if capturing:
            raise RuntimeError(
                "CUDA graph capture requires eight preallocated gradient tensors "
                "passed through out="
            )
        return tuple(
            torch.empty(shape, dtype=dtype, device=q.device)
            for _, shape, dtype in expected
        )
    if len(out) != len(expected):
        raise ValueError(f"out must contain eight tensors, got {len(out)}")
    result = tuple(out)
    for tensor, (name, shape, dtype) in zip(result, expected, strict=True):
        _validate_tensor(
            f"out[{name}]",
            tensor,
            shape=tuple(shape),
            dtype=dtype,
            device=q.device,
        )
    return result


def _allocate_low_workspace(
    workspace: _RecurrentKDABackwardWorkspaceBase,
    spec: _ShapeSpec,
    *,
    capturing: bool,
) -> tuple[torch.Tensor, ...]:
    token_vector_shape = (spec.total_tokens, spec.num_heads, _HEAD_DIM)
    token_scalar_shape = (spec.total_tokens, spec.num_heads)
    checkpoint_shape = (
        spec.total_tokens,
        spec.num_heads,
        _HEAD_DIM,
        _HEAD_DIM,
    )
    return (
        _buffer(
            workspace,
            "low_q_norm",
            token_vector_shape,
            torch.float32,
            capturing=capturing,
        ),
        _buffer(
            workspace,
            "low_k_norm",
            token_vector_shape,
            torch.float32,
            capturing=capturing,
        ),
        _buffer(
            workspace,
            "low_decay",
            token_vector_shape,
            torch.float32,
            capturing=capturing,
        ),
        _buffer(
            workspace,
            "low_beta_active",
            token_scalar_shape,
            torch.float32,
            capturing=capturing,
        ),
        _buffer(
            workspace,
            "low_checkpoint",
            checkpoint_shape,
            torch.float32,
            capturing=capturing,
        ),
        _buffer(
            workspace,
            "low_dq_normalized",
            token_vector_shape,
            torch.float32,
            capturing=capturing,
        ),
        _buffer(
            workspace,
            "low_dk_normalized",
            token_vector_shape,
            torch.float32,
            capturing=capturing,
        ),
        _buffer(
            workspace,
            "low_dlog_decay",
            token_vector_shape,
            torch.float32,
            capturing=capturing,
        ),
        _buffer(
            workspace,
            "low_dbeta_active",
            token_scalar_shape,
            torch.float32,
            capturing=capturing,
        ),
    )


def _allocate_high_workspace(
    workspace: _RecurrentKDABackwardWorkspaceBase,
    spec: _ShapeSpec,
    beta: torch.Tensor,
    *,
    capturing: bool,
) -> dict[str, torch.Tensor]:
    total_chunks = sum(
        (length + _CHUNK_SIZE - 1) // _CHUNK_SIZE for length in spec.seq_lens
    )
    h = spec.num_heads
    token_vector_shape = (spec.total_tokens, h, _HEAD_DIM)
    token_scalar_shape = (spec.total_tokens, h)
    tape_vector_shape = (total_chunks, h, _CHUNK_SIZE, _HEAD_DIM)
    tape_value_shape = (total_chunks, h, _HEAD_DIM, _CHUNK_SIZE)
    chunk_state_shape = (total_chunks, h, _HEAD_DIM, _HEAD_DIM)
    beta_tma = beta.reshape(spec.total_tokens, h)
    if spec.total_tokens < 32 or h < 8:
        beta_tma = _buffer(
            workspace,
            "high_beta_tma",
            (max(spec.total_tokens, 32), max(h, 8)),
            torch.bfloat16,
            capturing=capturing,
        )

    def high_buffer(
        name: str, shape: tuple[int, ...], dtype: torch.dtype
    ) -> torch.Tensor:
        return _buffer(
            workspace,
            f"high_{name}",
            shape,
            dtype,
            capturing=capturing,
        )

    tensors = {
        "beta_tma": beta_tma,
        "q_tma_out": high_buffer("q_tma_out", token_vector_shape, torch.bfloat16),
        "forward_final": high_buffer(
            "forward_final",
            (spec.num_sequences, h, _HEAD_DIM, _HEAD_DIM),
            torch.bfloat16,
        ),
        "chunk_state": high_buffer("chunk_state", chunk_state_shape, torch.bfloat16),
        "state_checkpoint_needed": high_buffer(
            "state_checkpoint_needed",
            ((total_chunks + spec.num_sequences) * h,),
            torch.uint32,
        ),
        "tape_qd": high_buffer("tape_qd", tape_vector_shape, torch.bfloat16),
        "tape_kd": high_buffer("tape_kd", tape_vector_shape, torch.bfloat16),
        "tape_kr": high_buffer("tape_kr", tape_vector_shape, torch.bfloat16),
        "tape_j": high_buffer(
            "tape_j",
            (total_chunks, h, _CHUNK_SIZE, _CHUNK_SIZE),
            torch.bfloat16,
        ),
        "tape_restore_factor": high_buffer(
            "tape_restore_factor",
            (total_chunks, h, _HEAD_DIM),
            torch.float32,
        ),
        "tape_x": high_buffer("tape_x", tape_value_shape, torch.bfloat16),
        "tape_r": high_buffer("tape_r", tape_value_shape, torch.bfloat16),
        "norm_inv": high_buffer("norm_inv", (spec.total_tokens, h, 2), torch.float32),
        "decay": high_buffer("decay", token_vector_shape, torch.bfloat16),
        "beta_active": high_buffer("beta_active", token_scalar_shape, torch.float32),
        "zero_workspace": high_buffer(
            "zero_workspace",
            (total_chunks * h,),
            torch.uint32,
        ),
        "chunk_dh": high_buffer("chunk_dh", chunk_state_shape, torch.bfloat16),
        "chunk_dr": high_buffer("chunk_dr", tape_value_shape, torch.bfloat16),
        "chunk_dx": high_buffer("chunk_dx", tape_value_shape, torch.bfloat16),
        "grad_qd": high_buffer("grad_qd", tape_vector_shape, torch.bfloat16),
        "grad_kd": high_buffer("grad_kd", tape_vector_shape, torch.bfloat16),
        "grad_ki": high_buffer("grad_ki", tape_vector_shape, torch.bfloat16),
        "dlog_decay": high_buffer("dlog_decay", token_vector_shape, torch.float32),
        "dbeta_active": high_buffer("dbeta_active", token_scalar_shape, torch.float32),
    }
    tensors["tape_e"] = tensors["tape_x"]
    return tensors


def _allocate_c16_workspace(
    workspace: _RecurrentKDABackwardWorkspaceBase,
    spec: _ShapeSpec,
    *,
    capturing: bool,
) -> dict[str, torch.Tensor]:
    total_chunks = sum(length // _C16_CHUNK_SIZE for length in spec.seq_lens)
    h = spec.num_heads
    token_vector_shape = (spec.total_tokens, h, _HEAD_DIM)
    token_scalar_shape = (spec.total_tokens, h)

    def c16_buffer(
        name: str, shape: tuple[int, ...], dtype: torch.dtype
    ) -> torch.Tensor:
        return _buffer(
            workspace,
            f"c16_{name}",
            shape,
            dtype,
            capturing=capturing,
        )

    return {
        "forward_out": c16_buffer("forward_out", token_vector_shape, torch.bfloat16),
        "forward_final_bf16": c16_buffer(
            "forward_final_bf16",
            (spec.num_sequences, h, _HEAD_DIM, _HEAD_DIM),
            torch.bfloat16,
        ),
        "forward_final": c16_buffer(
            "forward_final",
            (spec.num_sequences, h, _HEAD_DIM, _HEAD_DIM),
            torch.float32,
        ),
        "state_checkpoints": c16_buffer(
            "state_checkpoints",
            (total_chunks, h, _HEAD_DIM, _HEAD_DIM),
            torch.bfloat16,
        ),
        "beta_active": c16_buffer("beta_active", token_scalar_shape, torch.bfloat16),
        "dlog_decay": c16_buffer("dlog_decay", token_vector_shape, torch.float32),
        "dlog_boundary": c16_buffer(
            "dlog_boundary", (total_chunks, h, _HEAD_DIM), torch.float32
        ),
        "dbeta_active": c16_buffer("dbeta_active", token_scalar_shape, torch.float32),
        "gate_part_a": c16_buffer("gate_part_a", (128, h, _HEAD_DIM), torch.float32),
        "gate_part_dt": c16_buffer("gate_part_dt", (128, h, _HEAD_DIM), torch.float32),
        "counter": c16_buffer("counter", (1,), torch.uint32),
        "dummy_u32": c16_buffer("dummy_u32", (1,), torch.uint32),
        "dummy_f32": c16_buffer("dummy_f32", (1,), torch.float32),
    }


def _get_flash_kda_backward_module(device: torch.device):
    from .jit.flash_kda_backward import (
        FlashKDABackwardTarget,
        get_flash_kda_backward_module,
    )

    capability = get_compute_capability(device)
    target: FlashKDABackwardTarget = "sm100a" if capability == (10, 0) else "sm103a"
    return get_flash_kda_backward_module(target)


def _get_flash_kda_training_module(device: torch.device):
    from .jit.flash_kda_training import (
        FlashKDATrainingTarget,
        load_flash_kda_training_module,
    )

    capability = get_compute_capability(device)
    target: FlashKDATrainingTarget = "sm100a" if capability == (10, 0) else "sm103a"
    return load_flash_kda_training_module(target)


@flashinfer_api
def recurrent_kda_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor,
    do: torch.Tensor,
    dfinal_state: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    lower_bound: float = _DEFAULT_LOWER_BOUND,
    workspace: Optional[RecurrentKDABackwardWorkspace] = None,
    out: Optional[Sequence[torch.Tensor]] = None,
) -> tuple[torch.Tensor, ...]:
    r"""Compute all gradients of the recurrent KDA training recurrence.

    The kernel differentiates BF16 Q/K/V/raw-gate/raw-beta token inputs and
    FP32 parameters and recurrent state. Q and K are L2-normalized with
    epsilon ``1e-6``; decay is
    ``exp(lower_bound * sigmoid(exp(A_log) * (g + dt_bias)))`` and beta is
    passed through a sigmoid. The loss adjoints are ``do`` for token output
    and ``dfinal_state`` for final recurrent state.

    This frozen implementation is specialized for SM100a/SM103a, head/key/value
    dimension 128, FP32 state, and the eight shapes listed in
    :doc:`../api/kda_backward`. It returns ``(dq, dk, dv, dg, dbeta,
    dA_log, ddt_bias, dinitial_state)`` in that order.

    Args:
        q: Contiguous BF16 ``[1,T,H,128]`` query tensor.
        k: Contiguous BF16 ``[1,T,H,128]`` key tensor.
        v: Contiguous BF16 ``[1,T,H,128]`` value tensor.
        g: Contiguous BF16 ``[1,T,H,128]`` raw gate tensor.
        beta: Contiguous BF16 ``[1,T,H]`` raw beta logits.
        A_log: Contiguous FP32 ``[H]`` decay parameter.
        dt_bias: Contiguous FP32 ``[H,128]`` decay bias.
        initial_state: Contiguous FP32 value-first ``[N,H,V,K]`` state, with
            ``V=K=128``.
        do: Contiguous BF16 token-output adjoint matching ``q``.
        dfinal_state: Contiguous FP32 value-first final-state adjoint matching
            ``initial_state``.
        cu_seqlens: Exact contiguous CUDA int64 packed
            offsets for one of the documented packed shapes, otherwise ``None``.
        scale: Fixed output scale ``1 / sqrt(128)``.
        lower_bound: Fixed safe-gate lower bound ``-5.0``.
        workspace: Reusable scratch.
            Required, eagerly warmed, for CUDA Graph capture.
        out: Eight preallocated gradient
            tensors in return order. Required for CUDA Graph capture.
    """

    spec = _validate_and_select_shape(
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        initial_state,
        do,
        dfinal_state,
        cu_seqlens,
    )
    scale_value = 1.0 / math.sqrt(_HEAD_DIM) if scale is None else float(scale)
    lower_bound_value = float(lower_bound)
    required_scale = 1.0 / math.sqrt(_HEAD_DIM)
    if not math.isfinite(scale_value) or abs(scale_value - required_scale) > 1e-15:
        raise ValueError(
            f"recurrent_kda_backward fixes scale=1/sqrt(128), got {scale_value}"
        )
    if not math.isfinite(lower_bound_value) or lower_bound_value != -5.0:
        raise ValueError(
            f"recurrent_kda_backward fixes lower_bound=-5.0, got {lower_bound_value}"
        )

    capturing = torch.cuda.is_current_stream_capturing()
    if capturing and workspace is None:
        raise RuntimeError(
            "CUDA graph capture requires an explicit "
            "RecurrentKDABackwardWorkspace warmed with the exact tensors"
        )
    outputs = _validate_outputs(
        out,
        q=q,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=initial_state,
        capturing=capturing,
    )
    stream_ptr = int(torch.cuda.current_stream(q.device).cuda_stream)
    explicit_workspace = workspace is not None
    active_workspace: _RecurrentKDABackwardWorkspaceBase
    if workspace is None:
        active_workspace = _get_stream_workspace(q.device, stream_ptr)
    else:
        active_workspace = workspace

    with active_workspace._lock:
        _bind_workspace(
            active_workspace,
            device=q.device,
            stream_ptr=stream_ptr,
            capturing=capturing,
            explicit=explicit_workspace,
        )
        _validate_packed_offsets(
            active_workspace,
            spec,
            cu_seqlens,
            capturing=capturing,
        )
        metadata = _prepare_metadata(active_workspace, spec, capturing=capturing)
        cu_seqlens_arg = cu_seqlens if spec.packed else metadata["fixed_cu_seqlens"]
        assert cu_seqlens_arg is not None
        launch_signature = (
            spec.name,
            scale_value,
            lower_bound_value,
            *(
                _tensor_signature(tensor)
                for tensor in (
                    q,
                    k,
                    v,
                    g,
                    beta,
                    A_log,
                    dt_bias,
                    initial_state,
                    do,
                    dfinal_state,
                    cu_seqlens_arg,
                    *outputs,
                )
            ),
        )
        if capturing and active_workspace._warmed_signature != launch_signature:
            raise RuntimeError(
                "RecurrentKDABackwardWorkspace is not warmed for the exact "
                "input, output, metadata, scale, and lower-bound signature"
            )

        module = _get_flash_kda_backward_module(q.device)
        dq, dk, dv, dg, dbeta, dA_log, ddt_bias, dinitial_state = outputs
        if spec.c16_route:
            c16 = _allocate_c16_workspace(
                active_workspace,
                spec,
                capturing=capturing,
            )
            forward_work_items = metadata["c16_forward_work_items"].view(-1, 8)
            backward_work_items = metadata["c16_backward_work_items"].view(-1, 5)
            forward_descriptor_signature = tuple(
                _tensor_signature(tensor)
                for tensor in (
                    q,
                    k,
                    v,
                    g,
                    c16["forward_out"],
                    c16["state_checkpoints"],
                )
            )
            descriptor_signature = tuple(
                _tensor_signature(tensor)
                for tensor in (
                    q,
                    k,
                    v,
                    g,
                    do,
                    c16["state_checkpoints"],
                    dv,
                    c16["beta_active"],
                )
            )
            if capturing:
                if (
                    active_workspace._forward_descriptor_signature
                    != forward_descriptor_signature
                    or active_workspace._descriptor_signature != descriptor_signature
                ):
                    raise RuntimeError(
                        "C16 TMA descriptors were not warmed for the exact tensor signature"
                    )
                prepare_forward_descriptors = 0
                prepare_backward_descriptors = 0
            else:
                prepare_forward_descriptors = int(
                    active_workspace._forward_descriptor_signature
                    != forward_descriptor_signature
                )
                prepare_backward_descriptors = int(
                    active_workspace._descriptor_signature != descriptor_signature
                )
            if (
                active_workspace._descriptor_storage.numel() < _DESCRIPTOR_BYTES
                or active_workspace._descriptor_storage.data_ptr()
                % _DESCRIPTOR_ALIGNMENT
                != 0
            ):
                raise RuntimeError("invalid C16 descriptor storage")
            try:
                _get_flash_kda_training_module(q.device).run_forward(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    A_log,
                    dt_bias,
                    initial_state,
                    cu_seqlens_arg,
                    metadata["c16_checkpoint_cu_starts"],
                    forward_work_items,
                    active_workspace._forward_descriptor_storage,
                    c16["forward_out"],
                    c16["forward_final_bf16"],
                    c16["forward_final"],
                    c16["state_checkpoints"],
                    c16["beta_active"],
                    c16["counter"],
                    prepare_forward_descriptors,
                    spec.num_sequences,
                    spec.num_heads,
                    scale_value,
                    lower_bound_value,
                    stream_ptr,
                )
                module.run_c16_backward(
                    q,
                    k,
                    v,
                    g,
                    A_log,
                    dt_bias,
                    do,
                    dfinal_state,
                    cu_seqlens_arg,
                    backward_work_items,
                    active_workspace._descriptor_storage,
                    c16["state_checkpoints"],
                    c16["beta_active"],
                    c16["dlog_decay"],
                    c16["dlog_boundary"],
                    c16["dbeta_active"],
                    c16["gate_part_a"],
                    c16["gate_part_dt"],
                    c16["counter"],
                    c16["dummy_u32"],
                    c16["dummy_f32"],
                    dq,
                    dk,
                    dv,
                    dg,
                    dbeta,
                    dA_log,
                    ddt_bias,
                    dinitial_state,
                    prepare_backward_descriptors,
                    spec.num_sequences,
                    spec.num_heads,
                    scale_value,
                    lower_bound_value,
                    stream_ptr,
                )
            except Exception:
                if prepare_forward_descriptors:
                    active_workspace._forward_descriptor_signature = None
                if prepare_backward_descriptors:
                    active_workspace._descriptor_signature = None
                raise
            if prepare_forward_descriptors:
                active_workspace._forward_descriptor_signature = (
                    forward_descriptor_signature
                )
            if prepare_backward_descriptors:
                active_workspace._descriptor_signature = descriptor_signature
        elif not spec.high_head_route:
            (
                q_norm,
                k_norm,
                decay,
                beta_active,
                checkpoint,
                dq_normalized,
                dk_normalized,
                dlog_decay,
                dbeta_active,
            ) = _allocate_low_workspace(active_workspace, spec, capturing=capturing)
            module.run_low(
                q,
                k,
                v,
                g,
                beta,
                A_log,
                dt_bias,
                initial_state,
                do,
                dfinal_state,
                cu_seqlens_arg,
                q_norm,
                k_norm,
                decay,
                beta_active,
                checkpoint,
                dq_normalized,
                dk_normalized,
                dlog_decay,
                dbeta_active,
                dq,
                dk,
                dv,
                dg,
                dbeta,
                dA_log,
                ddt_bias,
                dinitial_state,
                spec.num_sequences,
                spec.num_heads,
                scale_value,
                lower_bound_value,
                stream_ptr,
            )
        else:
            high = _allocate_high_workspace(
                active_workspace,
                spec,
                beta,
                capturing=capturing,
            )
            descriptor_signature = tuple(
                _tensor_signature(tensor)
                for tensor in (
                    q,
                    k,
                    v,
                    g,
                    high["beta_tma"],
                    high["q_tma_out"],
                )
            )
            if capturing:
                if active_workspace._descriptor_signature != descriptor_signature:
                    raise RuntimeError(
                        "high-head TMA descriptors were not warmed for the exact "
                        "tensor signature"
                    )
                prepare_descriptors = 0
            else:
                prepare_descriptors = int(
                    active_workspace._descriptor_signature != descriptor_signature
                )
            if (
                active_workspace._descriptor_storage.numel() < _DESCRIPTOR_BYTES
                or active_workspace._descriptor_storage.data_ptr()
                % _DESCRIPTOR_ALIGNMENT
                != 0
            ):
                raise RuntimeError("invalid high-head descriptor storage")
            try:
                module.run_high(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    high["beta_tma"],
                    A_log,
                    dt_bias,
                    initial_state,
                    do,
                    dfinal_state,
                    cu_seqlens_arg,
                    metadata["seq_order"],
                    metadata["cu_chunk_offsets"],
                    metadata["consumer_chunk_order"],
                    metadata["chunk_sequence"],
                    metadata["chunk_index"],
                    metadata["chunk_pair_start"],
                    active_workspace._descriptor_storage,
                    high["q_tma_out"],
                    high["forward_final"],
                    high["chunk_state"],
                    high["state_checkpoint_needed"],
                    high["tape_qd"],
                    high["tape_kd"],
                    high["tape_kr"],
                    high["tape_j"],
                    high["tape_restore_factor"],
                    high["tape_e"],
                    high["tape_x"],
                    high["tape_r"],
                    high["norm_inv"],
                    high["decay"],
                    high["beta_active"],
                    high["zero_workspace"],
                    high["chunk_dh"],
                    high["chunk_dr"],
                    high["chunk_dx"],
                    high["grad_qd"],
                    high["grad_kd"],
                    high["grad_ki"],
                    high["dlog_decay"],
                    high["dbeta_active"],
                    dq,
                    dk,
                    dv,
                    dg,
                    dbeta,
                    dA_log,
                    ddt_bias,
                    dinitial_state,
                    prepare_descriptors,
                    spec.num_sequences,
                    spec.num_heads,
                    scale_value,
                    lower_bound_value,
                    stream_ptr,
                )
            except Exception:
                if prepare_descriptors:
                    active_workspace._descriptor_signature = None
                raise
            if prepare_descriptors:
                active_workspace._descriptor_signature = descriptor_signature

        if not capturing:
            active_workspace._warmed_signature = launch_signature
        elif explicit_workspace:
            active_workspace._captured = True
    return outputs


__all__ = [
    "KDA_BACKWARD_GRADIENT_NAMES",
    "RecurrentKDABackwardWorkspace",
    "recurrent_kda_backward",
]
