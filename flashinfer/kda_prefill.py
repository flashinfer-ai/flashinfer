"""
Copyright (c) 2025 by FlashInfer team.

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

"""
Kimi Delta Attention Prefill - Backend Layer
============================================

This file provides workspace management, validation, and frozen-kernel launch
support for recurrent KDA prefill.  The stable public dispatcher remains in
``flashinfer.kda``.
"""

import functools
import math
import threading
from typing import TYPE_CHECKING, Literal, Optional

import torch

from .utils import get_compute_capability

if TYPE_CHECKING:
    from .jit.flash_kda import FlashKDATarget, FlashKDAVariant

_FLASH_KDA_HEAD_DIM = 128
_FLASH_KDA_BETA_TMA_HEADS_PER_BOX = 8
_FLASH_KDA_SUPPORTED_COMPUTE_CAPABILITIES = {(10, 0), (10, 3)}
_FLASH_KDA_DESCRIPTOR_STORAGE_BYTES = 6 * 128
_FLASH_KDA_SMALL_BH_DESCRIPTOR_STORAGE_BYTES = 7 * 128
_FLASH_KDA_PERSISTENT_MIN_BALANCED_CTAS = 128
_FLASH_KDA_LPT_MAX_IMBALANCE_NUMERATOR = 21
_FLASH_KDA_LPT_MAX_IMBALANCE_DENOMINATOR = 20
_FLASH_KDA_GB200_LPT_MAX_IMBALANCE_NUMERATOR = 263
_FLASH_KDA_GB200_LPT_MAX_IMBALANCE_DENOMINATOR = 250
_FLASH_KDA_SMALL_BH_GROUP_SIZE = 8
_FLASH_KDA_SMALL_BH_RING_STAGES = 35
_FLASH_KDA_SMALL_BH_PACKET_ROWS = 123
_FLASH_KDA_SMALL_BH_PACKET_ELEMENTS = 128
_FLASH_KDA_SMALL_BH_MAX_TASKS = 8
_FLASH_KDA_SMALL_BH_MIN_SEQUENCE_LENGTH = 2048
_flash_kda_tensor_cache: dict[tuple, torch.Tensor] = {}
_flash_kda_tensor_cache_lock = threading.Lock()

_PackedMetadataSignature = tuple[int, int, int, int, bool]
_PersistentTaskPlan = tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
_PackedTaskMetadata = tuple[tuple[int, ...], Optional[_PersistentTaskPlan], bool]


class _RecurrentKDAPrefillWorkspaceBase:
    def __init__(self, device: torch.device | str) -> None:
        normalized_device = torch.device(device)
        if normalized_device.type != "cuda":
            raise ValueError("RecurrentKDAPrefillWorkspace requires a CUDA device")
        if normalized_device.index is None:
            normalized_device = torch.device("cuda", torch.cuda.current_device())
        self.device = normalized_device
        self._lock = threading.Lock()
        self._state_scratch: Optional[torch.Tensor] = None
        self._beta_padding: Optional[torch.Tensor] = None
        self._small_bh_packet_workspace: Optional[torch.Tensor] = None
        self._small_bh_packet_ready: Optional[torch.Tensor] = None
        self._small_bh_packet_consumed: Optional[torch.Tensor] = None
        self._small_bh_helper_done: Optional[torch.Tensor] = None
        self._descriptor_storages = {
            variant: torch.empty(
                (
                    _FLASH_KDA_SMALL_BH_DESCRIPTOR_STORAGE_BYTES
                    if variant == "small_bh_m128"
                    else _FLASH_KDA_DESCRIPTOR_STORAGE_BYTES
                ),
                dtype=torch.uint8,
                device=self.device,
            )
            for variant in (
                "m64",
                "m128",
                "m128_n16",
                "persistent_m128",
                "small_bh_m128",
            )
        }
        self._descriptor_signatures: dict[str, tuple] = {}
        self._packed_metadata_lock = threading.Lock()
        self._packed_metadata_tensor: Optional[torch.Tensor] = None
        self._packed_metadata_signature: Optional[_PackedMetadataSignature] = None
        self._packed_metadata: Optional[_PackedTaskMetadata] = None
        self._bound_stream_ptr: Optional[int] = None
        self._captured = False


class RecurrentKDAPrefillWorkspace(_RecurrentKDAPrefillWorkspaceBase):
    """Caller-owned storage required for recurrent-KDA CUDA graph capture.

    Construct one workspace per captured
    :func:`flashinfer.kda.recurrent_kda` invocation on the graph's CUDA
    device. Warm it by invoking that function eagerly with the exact tensors
    and capture stream, then synchronize that stream before capture. The
    workspace owns optional final-state scratch for calls without an initial
    state, beta padding, and schedule-specific M64/M128-N32/M128-N16 TMA
    descriptor storage and small-BH packet-ring storage for the lifetime of
    the graph. Persistent M128 is an
    eager-only B200/GB200 route; explicit workspaces use direct M128 or M64 so graph
    capture never synchronizes sequence lengths to construct host task bins.

    A workspace binds to its first stream. Once it participates in capture it
    cannot be passed to Python again, either eagerly or in another capture.
    Graph replay does not invoke Python and remains valid for the lifetime of
    the workspace.
    """


class _FlashKDAStreamWorkspace(_RecurrentKDAPrefillWorkspaceBase):
    """Internal eager-only workspace for one CUDA stream."""


_flash_kda_stream_workspaces: dict[tuple[int, int], _FlashKDAStreamWorkspace] = {}
_flash_kda_stream_workspaces_lock = threading.Lock()


def _is_plain_multi_token_prefill(
    q: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor],
    num_spec_tokens: Optional[int],
) -> bool:
    if num_spec_tokens is not None or not isinstance(q, torch.Tensor) or q.ndim != 4:
        return False
    if cu_seqlens is None:
        return q.shape[1] > 1
    if not isinstance(cu_seqlens, torch.Tensor) or cu_seqlens.ndim != 1:
        return False
    num_sequences = cu_seqlens.numel() - 1
    return num_sequences > 0 and q.shape[1] > num_sequences


def _is_contiguous_cuda_tensor(
    tensor: Optional[torch.Tensor],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> bool:
    return (
        isinstance(tensor, torch.Tensor)
        and tensor.is_cuda
        and tensor.device == device
        and tensor.dtype == dtype
        and tensor.is_contiguous()
    )


def _is_token_row_strided_cuda_tensor(
    tensor: Optional[torch.Tensor],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> bool:
    return (
        isinstance(tensor, torch.Tensor)
        and tensor.is_cuda
        and tensor.device == device
        and tensor.dtype == dtype
        and tensor.ndim >= 2
        and tensor.stride(-1) == 1
        and tensor.stride(-2) >= tensor.shape[-1]
    )


def _is_state_pool_tensor(
    tensor: Optional[torch.Tensor],
    *,
    device: torch.device,
    num_heads: int,
) -> bool:
    return (
        isinstance(tensor, torch.Tensor)
        and tensor.is_cuda
        and tensor.device == device
        and tensor.dtype == torch.bfloat16
        and tensor.ndim == 4
        and tensor.shape[0] > 0
        and tensor.data_ptr() % 16 == 0
        and tuple(tensor.shape[1:])
        == (num_heads, _FLASH_KDA_HEAD_DIM, _FLASH_KDA_HEAD_DIM)
        and tensor.stride(-1) == 1
        and tensor.stride(-2) == _FLASH_KDA_HEAD_DIM
        and tensor.stride(-3) == _FLASH_KDA_HEAD_DIM * _FLASH_KDA_HEAD_DIM
        and tensor.stride(0) >= num_heads * _FLASH_KDA_HEAD_DIM * _FLASH_KDA_HEAD_DIM
        and tensor.stride(0) * tensor.element_size() % 16 == 0
    )


def _flash_kda_prefill_is_eligible(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
    initial_state: Optional[torch.Tensor],
    use_qk_l2norm_in_kernel: bool,
    use_gate_in_kernel: bool,
    lower_bound: Optional[float],
    cu_seqlens: Optional[torch.Tensor],
    ssm_state_indices: Optional[torch.Tensor],
    num_spec_tokens: Optional[int],
    num_accepted_tokens: Optional[torch.Tensor],
    output: Optional[torch.Tensor],
    initial_state_source: Optional[torch.Tensor],
    initial_state_indices: Optional[torch.Tensor],
    beta_is_logit: bool,
    state_checkpoints: Optional[torch.Tensor],
    checkpoint_cu_starts: Optional[torch.Tensor],
    checkpoint_every_n_tokens: int,
) -> bool:
    """Return whether the call exactly matches the frozen FlashKDA contract."""

    if not _is_plain_multi_token_prefill(q, cu_seqlens, num_spec_tokens):
        return False
    if (
        num_accepted_tokens is not None
        or initial_state_source is not None
        or initial_state_indices is not None
    ):
        return False
    if not (
        use_qk_l2norm_in_kernel
        and use_gate_in_kernel
        and beta_is_logit
        and lower_bound is not None
        and math.isfinite(float(lower_bound))
        and float(lower_bound) < 0.0
    ):
        return False
    if (
        not q.is_cuda
        or get_compute_capability(q.device)
        not in _FLASH_KDA_SUPPORTED_COMPUTE_CAPABILITIES
    ):
        return False
    if not _is_contiguous_cuda_tensor(q, dtype=torch.bfloat16, device=q.device):
        return False
    if q.ndim != 4:
        return False
    batch_size, total_or_fixed_tokens, num_heads, head_dim = q.shape
    if (
        batch_size <= 0
        or total_or_fixed_tokens <= 1
        or num_heads <= 0
        or head_dim != _FLASH_KDA_HEAD_DIM
    ):
        return False
    for tensor in (k, v, g):
        if (
            not _is_contiguous_cuda_tensor(
                tensor, dtype=torch.bfloat16, device=q.device
            )
            or tensor.shape != q.shape
        ):
            return False
    if not _is_token_row_strided_cuda_tensor(
        beta, dtype=torch.bfloat16, device=q.device
    ) or beta.shape != (batch_size, total_or_fixed_tokens, num_heads):
        return False
    if batch_size > 1 and beta.stride(0) != total_or_fixed_tokens * beta.stride(1):
        return False
    if not _is_contiguous_cuda_tensor(
        A_log, dtype=torch.float32, device=q.device
    ) or A_log.shape != (num_heads,):
        return False
    if not _is_contiguous_cuda_tensor(dt_bias, dtype=torch.float32, device=q.device):
        return False
    if dt_bias.numel() != num_heads * _FLASH_KDA_HEAD_DIM or dt_bias.ndim not in (1, 2):
        return False
    if dt_bias.ndim == 2 and dt_bias.shape != (num_heads, _FLASH_KDA_HEAD_DIM):
        return False

    if cu_seqlens is None:
        num_sequences = batch_size
    else:
        if (
            batch_size != 1
            or not cu_seqlens.is_cuda
            or cu_seqlens.device != q.device
            or cu_seqlens.dtype not in (torch.int32, torch.int64)
            or cu_seqlens.ndim != 1
            or not cu_seqlens.is_contiguous()
        ):
            return False
        num_sequences = cu_seqlens.numel() - 1
        if num_sequences <= 0 or total_or_fixed_tokens <= num_sequences:
            return False

    if ssm_state_indices is not None:
        if (
            initial_state is None
            or not _is_contiguous_cuda_tensor(
                ssm_state_indices, dtype=torch.int32, device=q.device
            )
            or ssm_state_indices.ndim != 1
            or ssm_state_indices.numel() != num_sequences
        ):
            return False
    if initial_state is not None:
        if not _is_state_pool_tensor(
            initial_state,
            device=q.device,
            num_heads=num_heads,
        ):
            return False
        if ssm_state_indices is None and initial_state.shape[0] != num_sequences:
            return False
    if (
        checkpoint_every_n_tokens < 0
        or checkpoint_every_n_tokens > torch.iinfo(torch.int32).max
        or checkpoint_every_n_tokens % 32 != 0
    ):
        return False
    if checkpoint_every_n_tokens:
        if (
            not _is_contiguous_cuda_tensor(
                state_checkpoints, dtype=torch.bfloat16, device=q.device
            )
            or state_checkpoints.ndim != 4
            or tuple(state_checkpoints.shape[1:])
            != (num_heads, _FLASH_KDA_HEAD_DIM, _FLASH_KDA_HEAD_DIM)
            or not _is_contiguous_cuda_tensor(
                checkpoint_cu_starts, dtype=torch.int64, device=q.device
            )
            or checkpoint_cu_starts.ndim != 1
            or checkpoint_cu_starts.numel() != num_sequences + 1
        ):
            return False
    elif state_checkpoints is not None or checkpoint_cu_starts is not None:
        return False
    if output is not None:
        if (
            not _is_contiguous_cuda_tensor(
                output, dtype=torch.bfloat16, device=q.device
            )
            or output.shape != q.shape
        ):
            return False
    return True


def _select_flash_kda_prefill_variant(
    *,
    fixed_layout: bool,
    num_sequences: int,
    num_heads: int,
    needs_direct_m128: bool = False,
    use_persistent_m128: bool = False,
    use_small_bh_m128: bool = False,
    use_exact_n16: bool = False,
) -> "FlashKDAVariant":
    if num_heads == 12 or use_exact_n16:
        return "m128_n16"
    if (
        not needs_direct_m128
        and fixed_layout
        and num_sequences == 1
        and num_heads == 64
    ):
        return "m64"
    if use_small_bh_m128:
        return "small_bh_m128"
    if use_persistent_m128:
        return "persistent_m128"
    return "m128"


@functools.cache
def _flash_kda_device_sm_count(device: torch.device) -> int:
    """Resolve and cache the physical SM count for one CUDA device."""

    return int(torch.cuda.get_device_properties(device).multi_processor_count)


def _uses_measured_sm100_persistent_policy(
    *,
    compute_capability: tuple[int, int],
    sm_count: int,
) -> bool:
    return compute_capability == (10, 0) and sm_count in (148, 152)


def _should_use_small_bh_owner_helper(
    *,
    compute_capability: tuple[int, int],
    sm_count: int,
    fixed_layout: bool,
    num_sequences: int,
    num_heads: int,
    sequence_length: int,
) -> bool:
    """Select the fixed small-BH region whose eight-CTA groups fully reside."""

    total_tasks = num_sequences * num_heads
    return (
        compute_capability in _FLASH_KDA_SUPPORTED_COMPUTE_CAPABILITIES
        and fixed_layout
        and 0 < total_tasks <= _FLASH_KDA_SMALL_BH_MAX_TASKS
        and num_heads <= _FLASH_KDA_SMALL_BH_MAX_TASKS
        and sequence_length >= _FLASH_KDA_SMALL_BH_MIN_SEQUENCE_LENGTH
        and _FLASH_KDA_SMALL_BH_GROUP_SIZE * total_tasks <= sm_count
    )


def _requires_exact_n16_recurrence(
    *,
    sm_count: int,
    fixed_layout: bool,
    num_sequences: int,
    num_heads: int,
    uniform_sequences: bool,
) -> bool:
    """Select the measured N16 graph for the 148-SM H96/N128 holdout."""

    return (
        sm_count == 148
        and not fixed_layout
        and num_sequences == 128
        and num_heads == 96
        and uniform_sequences
    )


def _uniform_persistent_worker_count(total_tasks: int, *, sm_count: int) -> int:
    if total_tasks <= 0 or sm_count <= 0:
        raise ValueError("total_tasks and sm_count must be positive")
    if total_tasks <= sm_count:
        return total_tasks
    trips = (total_tasks + sm_count - 1) // sm_count
    if total_tasks % trips == 0:
        balanced_workers = total_tasks // trips
        if balanced_workers >= _FLASH_KDA_PERSISTENT_MIN_BALANCED_CTAS:
            return balanced_workers
    return sm_count


def _make_uniform_head_grouped_bins(
    *,
    num_sequences: int,
    num_heads: int,
    worker_count: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    total_tasks = num_sequences * num_heads
    if num_sequences <= 0 or num_heads <= 0 or not 0 < worker_count <= total_tasks:
        raise ValueError("head-grouped bins require positive sequence/head/task counts")
    task_ids: list[int] = []
    task_offsets = [0]
    for worker_idx in range(worker_count):
        begin = worker_idx * total_tasks // worker_count
        end = (worker_idx + 1) * total_tasks // worker_count
        for head_major_idx in range(begin, end):
            head_idx, ordered_seq_idx = divmod(head_major_idx, num_sequences)
            task_ids.append(ordered_seq_idx * num_heads + head_idx)
        task_offsets.append(len(task_ids))
    return tuple(task_ids), tuple(task_offsets)


def _make_lpt_task_bins(
    ordered_sequence_lengths: tuple[int, ...],
    *,
    num_heads: int,
    sm_count: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    total_tasks = len(ordered_sequence_lengths) * num_heads
    if (
        not ordered_sequence_lengths
        or num_heads <= 0
        or not 0 < sm_count <= total_tasks
    ):
        raise ValueError("LPT bins require positive sequence/head/task counts")
    bins: list[list[int]] = [[] for _ in range(sm_count)]
    loads = [0] * sm_count
    for ordered_seq_idx, seq_len in enumerate(ordered_sequence_lengths):
        chunk_count = (seq_len + 31) // 32
        for head_idx in range(num_heads):
            worker_idx = min(range(sm_count), key=lambda index: (loads[index], index))
            bins[worker_idx].append(ordered_seq_idx * num_heads + head_idx)
            loads[worker_idx] += chunk_count
    task_ids: list[int] = []
    task_offsets = [0]
    for worker_tasks in bins:
        task_ids.extend(worker_tasks)
        task_offsets.append(len(task_ids))
    return tuple(task_ids), tuple(task_offsets), tuple(loads)


def _lpt_bins_are_balanced(loads: tuple[int, ...]) -> bool:
    return bool(loads) and (
        max(loads) * _FLASH_KDA_LPT_MAX_IMBALANCE_DENOMINATOR * len(loads)
        <= sum(loads) * _FLASH_KDA_LPT_MAX_IMBALANCE_NUMERATOR
    )


def _persistent_task_plan(
    sequence_lengths: tuple[int, ...],
    *,
    num_heads: int,
    sm_count: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]] | None:
    """Build Cake's measured SM100 task plan, or return direct-route evidence."""

    total_tasks = len(sequence_lengths) * num_heads
    if sm_count not in (148, 152) or num_heads == 12 or total_tasks <= sm_count:
        return None
    sequence_order = tuple(
        sorted(
            range(len(sequence_lengths)),
            key=lambda index: sequence_lengths[index],
            reverse=True,
        )
    )
    ordered_lengths = tuple(sequence_lengths[index] for index in sequence_order)
    if len(set(sequence_lengths)) == 1:
        if num_heads not in (64, 96):
            return None
        worker_count = _uniform_persistent_worker_count(
            total_tasks,
            sm_count=sm_count,
        )
        task_ids, task_offsets = _make_uniform_head_grouped_bins(
            num_sequences=len(sequence_lengths),
            num_heads=num_heads,
            worker_count=worker_count,
        )
        return sequence_order, task_ids, task_offsets
    if num_heads != 96:
        return None
    task_ids, task_offsets, loads = _make_lpt_task_bins(
        ordered_lengths,
        num_heads=num_heads,
        sm_count=sm_count,
    )
    if sm_count == 152:
        if not loads or (
            max(loads) * _FLASH_KDA_GB200_LPT_MAX_IMBALANCE_DENOMINATOR * len(loads)
            > sum(loads) * _FLASH_KDA_GB200_LPT_MAX_IMBALANCE_NUMERATOR
        ):
            return None
    elif not _lpt_bins_are_balanced(loads):
        return None
    return sequence_order, task_ids, task_offsets


def _cached_tensor(
    key: tuple,
    factory,
    *,
    capture_error: str,
) -> torch.Tensor:
    with _flash_kda_tensor_cache_lock:
        tensor = _flash_kda_tensor_cache.get(key)
        if tensor is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(capture_error)
            tensor = factory()
            _flash_kda_tensor_cache[key] = tensor
        return tensor


def _cached_int32_metadata(
    *,
    device: torch.device,
    kind: str,
    values: tuple[int, ...],
) -> torch.Tensor:
    key = (kind, *_stream_cache_key(device), values)
    return _cached_tensor(
        key,
        lambda: torch.tensor(values, dtype=torch.int32, device=device),
        capture_error=(
            f"recurrent_kda {kind} metadata is not warmed for CUDA graph "
            "capture; invoke the same shape once before capture"
        ),
    )


def _fixed_cu_seqlens(
    *,
    device: torch.device,
    batch_size: int,
    seq_len: int,
) -> torch.Tensor:
    key = ("fixed_cu", *_stream_cache_key(device), batch_size, seq_len)
    return _cached_tensor(
        key,
        lambda: torch.arange(
            0,
            batch_size * seq_len + 1,
            seq_len,
            dtype=torch.int64,
            device=device,
        ),
        capture_error=(
            "fixed-layout recurrent_kda prefill metadata is not warmed for "
            "CUDA graph capture; invoke the same shape once before capture"
        ),
    )


def _identity_seq_order(
    *,
    device: torch.device,
    num_sequences: int,
) -> torch.Tensor:
    key = ("seq_order", *_stream_cache_key(device), num_sequences)
    return _cached_tensor(
        key,
        lambda: torch.arange(num_sequences, dtype=torch.int32, device=device),
        capture_error=(
            "recurrent_kda prefill seq_order is not warmed for CUDA graph "
            "capture; pass a preallocated seq_order or warm the shape first"
        ),
    )


def _dummy_bf16(device: torch.device) -> torch.Tensor:
    key = ("dummy_bf16", *_stream_cache_key(device))
    return _cached_tensor(
        key,
        lambda: torch.empty(1, dtype=torch.bfloat16, device=device),
        capture_error=(
            "recurrent_kda prefill dummy state is not warmed for CUDA graph "
            "capture; invoke the same device once before capture"
        ),
    )


def _dummy_i32(device: torch.device) -> torch.Tensor:
    key = ("dummy_i32", *_stream_cache_key(device))
    return _cached_tensor(
        key,
        lambda: torch.empty(1, dtype=torch.int32, device=device),
        capture_error=(
            "recurrent_kda prefill dummy int32 metadata is not warmed for "
            "CUDA graph capture; invoke the same device once before capture"
        ),
    )


def _dummy_i64(device: torch.device) -> torch.Tensor:
    key = ("dummy_i64", *_stream_cache_key(device))
    return _cached_tensor(
        key,
        lambda: torch.empty(1, dtype=torch.int64, device=device),
        capture_error=(
            "recurrent_kda prefill dummy int64 metadata is not warmed for "
            "CUDA graph capture; invoke the same device once before capture"
        ),
    )


def _stream_cache_key(device: torch.device) -> tuple[int, int]:
    stream = torch.cuda.current_stream(device)
    device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    return device_index, int(stream.cuda_stream)


def _get_stream_workspace(device: torch.device) -> _FlashKDAStreamWorkspace:
    key = _stream_cache_key(device)
    with _flash_kda_stream_workspaces_lock:
        workspace = _flash_kda_stream_workspaces.get(key)
        if workspace is None:
            workspace = _FlashKDAStreamWorkspace(device)
            _flash_kda_stream_workspaces[key] = workspace
        return workspace


def _cached_packed_task_metadata(
    workspace: _FlashKDAStreamWorkspace,
    cu_seqlens: torch.Tensor,
    *,
    total_tokens: int,
    num_heads: int,
    sm_count: int,
    build_persistent_plan: bool,
) -> _PackedTaskMetadata:
    """Cache host-built sequence order and optional persistent task bins."""

    signature = (
        int(cu_seqlens._version),
        total_tokens,
        num_heads,
        sm_count,
        build_persistent_plan,
    )
    with workspace._packed_metadata_lock:
        cached_metadata = workspace._packed_metadata
        if (
            workspace._packed_metadata_tensor is cu_seqlens
            and workspace._packed_metadata_signature == signature
            and cached_metadata is not None
        ):
            return cached_metadata
        offsets = tuple(int(value) for value in cu_seqlens.tolist())
        if (
            not offsets
            or offsets[0] != 0
            or offsets[-1] != total_tokens
            or any(
                right <= left for left, right in zip(offsets, offsets[1:], strict=False)
            )
        ):
            raise ValueError(
                "cu_seqlens must start at zero, be strictly increasing, "
                "and end at the packed token count"
            )
        sequence_lengths = tuple(
            right - left for left, right in zip(offsets, offsets[1:], strict=False)
        )
        sequence_order = tuple(
            sorted(
                range(len(sequence_lengths)),
                key=lambda index: sequence_lengths[index],
                reverse=True,
            )
        )
        persistent_plan = (
            _persistent_task_plan(
                sequence_lengths,
                num_heads=num_heads,
                sm_count=sm_count,
            )
            if build_persistent_plan
            else None
        )
        metadata = (
            sequence_order,
            persistent_plan,
            len(set(sequence_lengths)) == 1,
        )
        workspace._packed_metadata_tensor = cu_seqlens
        workspace._packed_metadata_signature = signature
        workspace._packed_metadata = metadata
        return metadata


def _workspace_buffer(
    *,
    workspace: _RecurrentKDAPrefillWorkspaceBase,
    attribute: str,
    device: torch.device,
    numel: int,
    capture_error: str,
    dtype: torch.dtype = torch.bfloat16,
    zero_on_allocate: bool = False,
) -> torch.Tensor:
    buffer = getattr(workspace, attribute)
    capturing = torch.cuda.is_current_stream_capturing()
    if buffer is None or buffer.numel() < numel:
        if capturing:
            raise RuntimeError(capture_error)
        factory = torch.zeros if zero_on_allocate else torch.empty
        buffer = factory(numel, dtype=dtype, device=device)
        setattr(workspace, attribute, buffer)
    elif buffer.dtype != dtype:
        raise RuntimeError(
            f"recurrent_kda workspace buffer {attribute} has dtype "
            f"{buffer.dtype}, expected {dtype}"
        )
    return buffer[:numel]


def _state_scratch(
    *,
    workspace: _RecurrentKDAPrefillWorkspaceBase,
    device: torch.device,
    shape: tuple[int, int, int, int],
) -> torch.Tensor:
    numel = math.prod(shape)
    return _workspace_buffer(
        workspace=workspace,
        attribute="_state_scratch",
        device=device,
        numel=numel,
        capture_error=(
            "recurrent_kda prefill final-state workspace is not large enough "
            "for CUDA graph capture; warm the largest shape on this stream "
            "before capture"
        ),
    ).view(shape)


def _beta_tma_source(
    beta: torch.Tensor,
    workspace: _RecurrentKDAPrefillWorkspaceBase,
) -> torch.Tensor:
    batch_size, seq_len, num_heads = beta.shape
    total_tokens = batch_size * seq_len
    if beta.stride(-1) != 1:
        raise ValueError("beta must have unit head stride")
    if batch_size == 1:
        beta_flat = beta[0]
    else:
        if beta.stride(0) != seq_len * beta.stride(1):
            raise ValueError("beta batch/token dimensions must collapse without a copy")
        beta_flat = beta.as_strided(
            (total_tokens, num_heads),
            (beta.stride(1), beta.stride(2)),
        )
    if (
        total_tokens >= 32
        and num_heads >= _FLASH_KDA_BETA_TMA_HEADS_PER_BOX
        and beta_flat.data_ptr() % 16 == 0
        and beta_flat.stride(0) * beta.element_size() % 16 == 0
    ):
        return beta_flat
    padded_tokens = max(total_tokens, 32)
    padded_heads = (
        (num_heads + _FLASH_KDA_BETA_TMA_HEADS_PER_BOX - 1)
        // _FLASH_KDA_BETA_TMA_HEADS_PER_BOX
        * _FLASH_KDA_BETA_TMA_HEADS_PER_BOX
    )
    shape = (padded_tokens, padded_heads)
    padded = _workspace_buffer(
        workspace=workspace,
        attribute="_beta_padding",
        device=beta.device,
        numel=math.prod(shape),
        capture_error=(
            "recurrent_kda prefill beta TMA workspace is not large enough for "
            "CUDA graph capture; warm the largest padded token/head shape on "
            "this stream before capture"
        ),
    ).view(shape)
    # The frozen binding refreshes head-padded storage from ``beta`` immediately
    # before launching the frozen kernel. Keeping pack + main-kernel submission
    # in one FFI call avoids two Python-dispatched activities and their host gap,
    # while retaining stable storage for the TMA descriptor and CUDA graphs.
    return padded


def _small_bh_workspace(
    *,
    workspace: _RecurrentKDAPrefillWorkspaceBase,
    device: torch.device,
    total_tasks: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    packet_slots = total_tasks * _FLASH_KDA_SMALL_BH_RING_STAGES
    packet_shape = (
        packet_slots * _FLASH_KDA_SMALL_BH_PACKET_ROWS,
        _FLASH_KDA_SMALL_BH_PACKET_ELEMENTS,
    )
    capture_error = (
        "recurrent_kda small-BH packet workspace is not large enough for "
        "CUDA graph capture; warm the largest small-BH shape on this stream "
        "before capture"
    )
    packet_workspace = _workspace_buffer(
        workspace=workspace,
        attribute="_small_bh_packet_workspace",
        device=device,
        numel=math.prod(packet_shape),
        capture_error=capture_error,
    ).view(packet_shape)
    packet_ready = _workspace_buffer(
        workspace=workspace,
        attribute="_small_bh_packet_ready",
        device=device,
        numel=packet_slots,
        capture_error=capture_error,
        dtype=torch.uint32,
        zero_on_allocate=True,
    )
    packet_consumed = _workspace_buffer(
        workspace=workspace,
        attribute="_small_bh_packet_consumed",
        device=device,
        numel=packet_slots,
        capture_error=capture_error,
        dtype=torch.uint32,
        zero_on_allocate=True,
    )
    helper_done = _workspace_buffer(
        workspace=workspace,
        attribute="_small_bh_helper_done",
        device=device,
        numel=total_tasks,
        capture_error=capture_error,
        dtype=torch.uint32,
        zero_on_allocate=True,
    )
    return packet_workspace, packet_ready, packet_consumed, helper_done


def _tensor_descriptor_signature(tensor: torch.Tensor) -> tuple:
    return (
        tensor.data_ptr(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
    )


def _descriptor_signature(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta_tma: torch.Tensor,
    out: torch.Tensor,
    packet_workspace: Optional[torch.Tensor] = None,
) -> tuple:
    signature = tuple(
        _tensor_descriptor_signature(tensor) for tensor in (q, k, v, g, beta_tma, out)
    )
    if packet_workspace is not None:
        signature += (_tensor_descriptor_signature(packet_workspace),)
    return signature


def _bind_workspace(
    workspace: _RecurrentKDAPrefillWorkspaceBase,
    *,
    device: torch.device,
    stream_ptr: int,
    capturing: bool,
    explicit: bool,
) -> None:
    if workspace.device != device:
        raise ValueError(
            "RecurrentKDAPrefillWorkspace is bound to "
            f"{workspace.device}, but recurrent_kda inputs are on {device}"
        )
    if workspace._bound_stream_ptr is None:
        workspace._bound_stream_ptr = stream_ptr
    elif workspace._bound_stream_ptr != stream_ptr:
        raise RuntimeError(
            "RecurrentKDAPrefillWorkspace is bound to a different CUDA "
            "stream; warm and capture it on one stream"
        )
    if explicit and workspace._captured:
        reuse_kind = "captured by another CUDA graph" if capturing else "reused eagerly"
        raise RuntimeError(
            "RecurrentKDAPrefillWorkspace has participated in CUDA graph "
            f"capture and cannot be {reuse_kind} or mutated"
        )


def _storage_ranges_overlap(
    left: torch.Tensor,
    right: torch.Tensor,
) -> bool:
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
    left_end = storage_end(left)
    right_end = storage_end(right)
    return left_start < right_end and right_start < left_end


def _check_output_does_not_overlap_inputs(
    output: torch.Tensor,
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor],
) -> None:
    for name, tensor in (
        ("q", q),
        ("k", k),
        ("v", v),
        ("g", g),
        ("beta", beta),
        ("initial_state", initial_state),
    ):
        if tensor is not None and _storage_ranges_overlap(output, tensor):
            raise ValueError(
                f"output must not overlap {name} for frozen recurrent_kda prefill"
            )


def _validate_prefill_seq_order(
    seq_order: Optional[torch.Tensor],
    *,
    fixed_layout: bool,
    num_sequences: int,
    device: torch.device,
) -> torch.Tensor:
    if seq_order is None:
        return _identity_seq_order(device=device, num_sequences=num_sequences)
    if fixed_layout:
        raise ValueError("seq_order is only supported for packed recurrent_kda prefill")
    if not isinstance(seq_order, torch.Tensor):
        raise TypeError("seq_order must be a torch.Tensor")
    if (
        not seq_order.is_cuda
        or seq_order.device != device
        or seq_order.dtype != torch.int32
        or seq_order.ndim != 1
        or not seq_order.is_contiguous()
        or seq_order.numel() != num_sequences
    ):
        raise ValueError(
            "seq_order must be a contiguous CUDA int32 tensor with one "
            f"entry per sequence ({num_sequences})"
        )
    return seq_order


def _is_cuda_version_at_least(version: str) -> bool:
    # Keep JIT imports lazy so importing the public KDA facade does not
    # initialize the extension toolchain.
    from .jit.cpp_ext import is_cuda_version_at_least

    return is_cuda_version_at_least(version)


def _select_flash_kda_prefill_target(device: torch.device) -> "FlashKDATarget":
    compute_capability = get_compute_capability(device)
    if compute_capability not in _FLASH_KDA_SUPPORTED_COMPUTE_CAPABILITIES:
        raise RuntimeError(
            "frozen recurrent-KDA prefill requires compute capability 10.0 "
            "(SM100a; B200/GB200) or 10.3 (SM103a; B300/GB300); got "
            f"{compute_capability[0]}.{compute_capability[1]}"
        )
    if compute_capability == (10, 0) and not _is_cuda_version_at_least("12.9"):
        if not _is_cuda_version_at_least("12.8"):
            raise RuntimeError(
                "frozen recurrent-KDA prefill on compute capability 10.0 "
                "requires CUDA 12.8 or newer"
            )
        return "sm100a"
    if not _is_cuda_version_at_least("12.9"):
        raise RuntimeError(
            "frozen recurrent-KDA prefill on compute capability 10.3 requires "
            "CUDA 12.9 or newer for the sm_100f family target"
        )
    return "sm100f"


def _get_flash_kda_prefill_module(variant: "FlashKDAVariant", target: "FlashKDATarget"):
    from .jit.flash_kda import get_flash_kda_prefill_module

    return get_flash_kda_prefill_module(variant, target)


def _run_flash_kda_prefill(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: Optional[float],
    initial_state: Optional[torch.Tensor],
    output_final_state: bool,
    lower_bound: float,
    cu_seqlens: Optional[torch.Tensor],
    output: Optional[torch.Tensor],
    seq_order: Optional[torch.Tensor],
    prefill_workspace: Optional[RecurrentKDAPrefillWorkspace],
    state_indices: Optional[torch.Tensor],
    state_checkpoints: Optional[torch.Tensor],
    checkpoint_cu_starts: Optional[torch.Tensor],
    checkpoint_every_n_tokens: int,
    backend: Literal["cake"] = "cake",
) -> (
    tuple[torch.Tensor, Optional[torch.Tensor]]
    | tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]
):
    if backend != "cake":
        raise ValueError(f"backend must be 'cake', got {backend!r}")
    capturing = torch.cuda.is_current_stream_capturing()
    if capturing and prefill_workspace is None:
        raise RuntimeError(
            "CUDA graph capture of recurrent_kda prefill requires an explicit "
            "RecurrentKDAPrefillWorkspace warmed with the exact tensors on "
            "the capture stream"
        )
    batch_size, seq_len, num_heads, _ = q.shape
    fixed_layout = cu_seqlens is None
    num_sequences = batch_size if fixed_layout else cu_seqlens.numel() - 1
    target = _select_flash_kda_prefill_target(q.device)
    compute_capability = get_compute_capability(q.device)
    sm_count = _flash_kda_device_sm_count(q.device)
    stream_workspace = (
        _get_stream_workspace(q.device) if prefill_workspace is None else None
    )
    needs_direct_m128 = (
        state_indices is not None
        or checkpoint_every_n_tokens != 0
        or not beta.is_contiguous()
        or seq_order is not None
        or (
            initial_state is not None
            and initial_state.stride(0)
            != num_heads * _FLASH_KDA_HEAD_DIM * _FLASH_KDA_HEAD_DIM
        )
    )
    small_bh_candidate = not needs_direct_m128 and _should_use_small_bh_owner_helper(
        compute_capability=compute_capability,
        sm_count=sm_count,
        fixed_layout=fixed_layout,
        num_sequences=num_sequences,
        num_heads=num_heads,
        sequence_length=seq_len,
    )
    persistent_candidate = (
        _uses_measured_sm100_persistent_policy(
            compute_capability=compute_capability,
            sm_count=sm_count,
        )
        and not needs_direct_m128
        and prefill_workspace is None
        and initial_state is not None
        and num_heads != 12
        and not (fixed_layout and num_sequences == 1 and num_heads == 64)
        and num_sequences * num_heads > sm_count
    )
    automatic_sequence_order = None
    persistent_plan = None
    uniform_sequences = False
    if (
        not fixed_layout
        and seq_order is None
        and prefill_workspace is None
        and not capturing
    ):
        assert cu_seqlens is not None
        assert stream_workspace is not None
        (
            automatic_sequence_order,
            persistent_plan,
            uniform_sequences,
        ) = _cached_packed_task_metadata(
            stream_workspace,
            cu_seqlens,
            total_tokens=batch_size * seq_len,
            num_heads=num_heads,
            sm_count=sm_count,
            build_persistent_plan=persistent_candidate,
        )
    elif persistent_candidate:
        assert fixed_layout
        persistent_plan = _persistent_task_plan(
            (seq_len,) * num_sequences,
            num_heads=num_heads,
            sm_count=sm_count,
        )
    use_exact_n16 = _requires_exact_n16_recurrence(
        sm_count=sm_count,
        fixed_layout=fixed_layout,
        num_sequences=num_sequences,
        num_heads=num_heads,
        uniform_sequences=uniform_sequences,
    )
    if use_exact_n16:
        persistent_plan = None
    variant = _select_flash_kda_prefill_variant(
        fixed_layout=fixed_layout,
        num_sequences=num_sequences,
        num_heads=num_heads,
        needs_direct_m128=needs_direct_m128,
        use_persistent_m128=persistent_plan is not None,
        use_small_bh_m128=small_bh_candidate,
        use_exact_n16=use_exact_n16,
    )
    if fixed_layout:
        cu_seqlens_i64 = _fixed_cu_seqlens(
            device=q.device, batch_size=batch_size, seq_len=seq_len
        )
    else:
        assert cu_seqlens is not None
        if cu_seqlens.dtype == torch.int32 and capturing:
            raise RuntimeError(
                "packed recurrent_kda prefill requires int64 cu_seqlens "
                "during CUDA graph capture; convert it before capture"
            )
        cu_seqlens_i64 = (
            cu_seqlens
            if cu_seqlens.dtype == torch.int64
            else cu_seqlens.to(torch.int64)
        )
    persistent_task_ids = None
    persistent_task_offsets = None
    if persistent_plan is None:
        if automatic_sequence_order is None:
            seq_order_i32 = _validate_prefill_seq_order(
                seq_order,
                fixed_layout=fixed_layout,
                num_sequences=num_sequences,
                device=q.device,
            )
        else:
            seq_order_i32 = _cached_int32_metadata(
                device=q.device,
                kind="automatic_seq_order",
                values=automatic_sequence_order,
            )
    else:
        sequence_order, task_ids, task_offsets = persistent_plan
        seq_order_i32 = _cached_int32_metadata(
            device=q.device,
            kind="persistent_seq_order",
            values=sequence_order,
        )
        persistent_task_ids = _cached_int32_metadata(
            device=q.device,
            kind="persistent_task_ids",
            values=task_ids,
        )
        persistent_task_offsets = _cached_int32_metadata(
            device=q.device,
            kind="persistent_task_offsets",
            values=task_offsets,
        )
    dummy_state = _dummy_bf16(q.device)
    dummy_i32 = _dummy_i32(q.device) if variant != "m64" else None
    dummy_i64 = _dummy_i64(q.device) if variant != "m64" else None

    if output is None:
        if capturing:
            raise RuntimeError(
                "CUDA graph capture requires a preallocated output tensor for "
                "recurrent_kda prefill"
            )
        out_buf = torch.empty_like(q)
    else:
        out_buf = output
    _check_output_does_not_overlap_inputs(
        out_buf,
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
    )

    state_shape = (
        num_sequences,
        num_heads,
        _FLASH_KDA_HEAD_DIM,
        _FLASH_KDA_HEAD_DIM,
    )
    use_initial_state = initial_state is not None
    use_state_indices = state_indices is not None
    if use_state_indices and initial_state is None:
        raise ValueError("state_indices requires an initial_state pool")
    if initial_state is not None:
        initial_state_arg = initial_state
        final_state_arg = initial_state
        store_final_state = True
        returned_state = initial_state
    elif output_final_state:
        initial_state_arg = dummy_state
        if prefill_workspace is None:
            final_state_arg = torch.empty(
                state_shape, dtype=torch.bfloat16, device=q.device
            )
            returned_state = final_state_arg
        else:
            # Assigned to caller-owned stable state scratch under its lock.
            returned_state = None
        store_final_state = True
    else:
        initial_state_arg = dummy_state
        final_state_arg = dummy_state
        store_final_state = False
        returned_state = None
    state_slot_stride = (
        initial_state.stride(0)
        if initial_state is not None and initial_state.ndim == 4
        else num_heads * _FLASH_KDA_HEAD_DIM * _FLASH_KDA_HEAD_DIM
    )

    scale_value = (
        1.0 / math.sqrt(_FLASH_KDA_HEAD_DIM) if scale is None else float(scale)
    )
    if not math.isfinite(scale_value):
        raise ValueError(f"scale must be finite, got {scale_value}")
    stream_ptr = int(torch.cuda.current_stream(q.device).cuda_stream)
    explicit_workspace = prefill_workspace is not None
    workspace: _RecurrentKDAPrefillWorkspaceBase
    if prefill_workspace is None:
        assert stream_workspace is not None
        workspace = stream_workspace
    else:
        workspace = prefill_workspace
    # TVM FFI may release the GIL. Serialize the complete shared-workspace
    # enqueue sequence so two host threads cannot interleave preparation or
    # launch on the same CUDA stream.
    with workspace._lock:
        _bind_workspace(
            workspace,
            device=q.device,
            stream_ptr=stream_ptr,
            capturing=capturing,
            explicit=explicit_workspace,
        )
        beta_tma = _beta_tma_source(beta, workspace)
        packet_workspace = None
        packet_ready = None
        packet_consumed = None
        helper_done = None
        if variant == "small_bh_m128":
            (
                packet_workspace,
                packet_ready,
                packet_consumed,
                helper_done,
            ) = _small_bh_workspace(
                workspace=workspace,
                device=q.device,
                total_tasks=num_sequences * num_heads,
            )
        if initial_state is None and output_final_state and explicit_workspace:
            final_state_arg = _state_scratch(
                workspace=workspace,
                device=q.device,
                shape=state_shape,
            )
            if initial_state is None:
                returned_state = final_state_arg
        signature = _descriptor_signature(
            q=q,
            k=k,
            v=v,
            g=g,
            beta_tma=beta_tma,
            out=out_buf,
            packet_workspace=packet_workspace,
        )
        warmed_signature = workspace._descriptor_signatures.get(variant)
        if capturing:
            if warmed_signature != signature:
                raise RuntimeError(
                    "RecurrentKDAPrefillWorkspace is not warmed for the exact "
                    f"{variant} descriptor signature; eagerly invoke the same "
                    "call on this stream before capture"
                )
            prepare_descriptors = 0
        else:
            prepare_descriptors = int(warmed_signature != signature)
        descriptor_storage = workspace._descriptor_storages[variant]
        module = _get_flash_kda_prefill_module(variant, target)
        try:
            if variant == "m64":
                module.run(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    beta_tma,
                    A_log,
                    dt_bias,
                    cu_seqlens_i64,
                    seq_order_i32,
                    initial_state_arg,
                    out_buf,
                    final_state_arg,
                    descriptor_storage,
                    prepare_descriptors,
                    num_heads,
                    int(use_initial_state),
                    int(store_final_state),
                    scale_value,
                    float(lower_bound),
                    stream_ptr,
                )
            elif variant == "small_bh_m128":
                assert packet_workspace is not None
                assert packet_ready is not None
                assert packet_consumed is not None
                assert helper_done is not None
                module.run(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    beta_tma,
                    A_log,
                    dt_bias,
                    cu_seqlens_i64,
                    seq_order_i32,
                    initial_state_arg,
                    out_buf,
                    final_state_arg,
                    descriptor_storage,
                    packet_workspace,
                    packet_ready,
                    packet_consumed,
                    helper_done,
                    prepare_descriptors,
                    num_heads,
                    int(use_initial_state),
                    int(store_final_state),
                    scale_value,
                    float(lower_bound),
                    stream_ptr,
                )
            elif variant == "persistent_m128":
                assert persistent_task_ids is not None
                assert persistent_task_offsets is not None
                module.run(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    beta_tma,
                    A_log,
                    dt_bias,
                    cu_seqlens_i64,
                    seq_order_i32,
                    persistent_task_ids,
                    persistent_task_offsets,
                    initial_state_arg,
                    out_buf,
                    final_state_arg,
                    descriptor_storage,
                    prepare_descriptors,
                    num_heads,
                    int(use_initial_state),
                    int(store_final_state),
                    scale_value,
                    float(lower_bound),
                    stream_ptr,
                )
            else:
                assert dummy_i32 is not None
                assert dummy_i64 is not None
                module.run(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    beta_tma,
                    A_log,
                    dt_bias,
                    cu_seqlens_i64,
                    seq_order_i32,
                    state_indices if state_indices is not None else dummy_i32,
                    initial_state_arg,
                    out_buf,
                    final_state_arg,
                    (
                        state_checkpoints
                        if state_checkpoints is not None
                        else dummy_state
                    ),
                    (
                        checkpoint_cu_starts
                        if checkpoint_cu_starts is not None
                        else dummy_i64
                    ),
                    descriptor_storage,
                    prepare_descriptors,
                    num_heads,
                    beta.stride(-2),
                    state_slot_stride,
                    int(use_state_indices),
                    int(use_initial_state),
                    int(store_final_state),
                    checkpoint_every_n_tokens,
                    scale_value,
                    float(lower_bound),
                    stream_ptr,
                )
        except Exception:
            if prepare_descriptors:
                workspace._descriptor_signatures.pop(variant, None)
            raise
        if prepare_descriptors:
            workspace._descriptor_signatures[variant] = signature
        if capturing and explicit_workspace:
            workspace._captured = True
    result = (out_buf, returned_state if output_final_state else None)
    if checkpoint_every_n_tokens:
        assert state_checkpoints is not None
        return (*result, state_checkpoints)
    return result
