"""Full-dispatch paired Blackwell recurrent KDA training implementation."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence, cast

import torch

from .api_logging import flashinfer_api
from .kda_backward import KDA_BACKWARD_GRADIENT_NAMES
from .utils import get_compute_capability

_HEAD_DIM = 128
_C16_CHUNK = 16
_C32_CHUNK = 32
_LOWER_BOUND = -5.0
_SCALE = 1.0 / math.sqrt(_HEAD_DIM)
_SUPPORTED_COMPUTE_CAPABILITIES = ((10, 0), (10, 3))
_FINAL_TMAP_SLOTS_PER_CTA = 10
_FINAL_TMAP_BYTES_PER_SLOT = 128
_FINAL_SHORT_GRID_CTAS = 128
_FINAL_LONG_GRID_CTAS = 148
_TrainingTarget = Literal["sm100a", "sm103a"]
_RouteTag = Literal["grouped_c16", "grouped_c32", "c16", "c32", "row_split"]
_RouteFamily = Literal["c16", "c32", "row_split"]


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
        offset = sum(
            (size - 1) * stride
            for size, stride in zip(tensor.shape, tensor.stride(), strict=True)
            if size > 0
        )
        return tensor.data_ptr() + (offset + 1) * tensor.element_size()

    return left.data_ptr() < storage_end(right) and right.data_ptr() < storage_end(left)


def _check_writes_do_not_overlap(
    writes: Sequence[tuple[str, torch.Tensor]],
    reads: Sequence[tuple[str, torch.Tensor]],
) -> None:
    for index, (write_name, write_tensor) in enumerate(writes):
        for read_name, read_tensor in reads:
            if _storage_ranges_overlap(write_tensor, read_tensor):
                raise ValueError(f"{write_name} must not overlap {read_name}")
        for other_name, other_tensor in writes[index + 1 :]:
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


@dataclass(frozen=True)
class _TrainingRouteSpec:
    tag: _RouteTag

    @property
    def family(self) -> _RouteFamily:
        if self.tag.endswith("c16"):
            return "c16"
        if self.tag.endswith("c32"):
            return "c32"
        return "row_split"

    @property
    def grouped(self) -> bool:
        return self.tag.startswith("grouped_")


def _select_training_route(
    seq_lens: tuple[int, ...], num_qk_heads: int, num_v_heads: int
) -> _TrainingRouteSpec:
    """Mirror the source dispatcher without promoting fast-route predicates to guards."""

    long_underfilled_c16 = (
        num_v_heads <= 8
        and max(seq_lens) >= 1024
        and all(length > 0 and length % _C16_CHUNK == 0 for length in seq_lens)
    )
    grouped = num_qk_heads != num_v_heads
    if grouped:
        return _TrainingRouteSpec(
            "grouped_c16" if long_underfilled_c16 else "grouped_c32"
        )
    if long_underfilled_c16:
        return _TrainingRouteSpec("c16")
    uniform_c16 = (
        num_v_heads >= 16
        and len(set(seq_lens)) == 1
        and seq_lens[0] > 0
        and seq_lens[0] % _C16_CHUNK == 0
    )
    if uniform_c16:
        chunks_per_item = seq_lens[0] // _C16_CHUNK
        work_items = len(seq_lens) * num_v_heads
        use_c16 = (
            num_v_heads <= 64
            and (
                (chunks_per_item <= 32 and work_items >= 128)
                or (chunks_per_item <= 64 and work_items >= 192)
            )
        ) or (
            num_v_heads > 64
            and (
                (chunks_per_item <= 64 and work_items >= 384)
                or (chunks_per_item <= 96 and work_items >= 768)
            )
        )
        if use_c16:
            return _TrainingRouteSpec("c16")
    return _TrainingRouteSpec("c32" if num_v_heads >= 16 else "row_split")


@dataclass(frozen=True)
class _TrainingShape:
    layout: Literal["fixed", "packed"]
    public_batch: int
    public_tokens: int
    total_tokens: int
    num_sequences: int
    num_qk_heads: int
    num_v_heads: int
    seq_lens: tuple[int, ...]
    offsets: tuple[int, ...]
    route: _TrainingRouteSpec


def _validate_forward_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor],
) -> _TrainingShape:
    if not isinstance(q, torch.Tensor) or not q.is_cuda:
        raise ValueError("recurrent_kda_training_forward requires CUDA tensors")
    device = q.device
    if get_compute_capability(device) not in _SUPPORTED_COMPUTE_CAPABILITIES:
        raise ValueError(
            "recurrent_kda_training_forward requires compute capability 10.0 or 10.3"
        )
    if q.ndim != 4 or q.shape[-1] != _HEAD_DIM:
        raise ValueError("q must have shape [B, T, num_qk_heads, 128]")
    batch, tokens, num_qk_heads = map(int, q.shape[:3])
    if batch <= 0 or tokens <= 0 or num_qk_heads <= 0:
        raise ValueError("batch, sequence length, and head count must be positive")
    if cu_seqlens is None:
        layout: Literal["fixed", "packed"] = "fixed"
        seq_lens = (tokens,) * batch
        offsets = tuple(index * tokens for index in range(batch + 1))
    else:
        layout = "packed"
        if batch != 1:
            raise ValueError("packed tensors must have physical batch dimension 1")
        if not isinstance(cu_seqlens, torch.Tensor) or cu_seqlens.ndim != 1:
            raise ValueError("cu_seqlens must be a one-dimensional CUDA tensor")
        _validate_tensor(
            cu_seqlens,
            "cu_seqlens",
            shape=(int(cu_seqlens.numel()),),
            dtype=torch.int64,
            device=device,
        )
        if cu_seqlens.numel() < 2:
            raise ValueError("cu_seqlens must contain at least one sequence")
        offsets = tuple(int(value) for value in cu_seqlens.detach().cpu().tolist())
        if offsets[0] != 0 or offsets[-1] != tokens:
            raise ValueError("cu_seqlens must start at zero and end at total_tokens")
        seq_lens = tuple(
            right - left for left, right in zip(offsets, offsets[1:], strict=False)
        )
    if any(length <= 0 for length in seq_lens):
        raise ValueError("all sequence lengths must be positive")
    if v.ndim != 4 or v.shape[:2] != q.shape[:2] or v.shape[-1] != _HEAD_DIM:
        raise ValueError("v must have shape [B, T, num_v_heads, 128]")
    num_v_heads = int(v.shape[2])
    if num_v_heads <= 0 or num_v_heads % num_qk_heads != 0:
        raise ValueError("num_v_heads must be an integer multiple of num_qk_heads")
    qk_shape = (batch, tokens, num_qk_heads, _HEAD_DIM)
    value_shape = (batch, tokens, num_v_heads, _HEAD_DIM)
    for name, tensor, expected in (
        ("q", q, qk_shape),
        ("k", k, qk_shape),
        ("v", v, value_shape),
        ("g", g, value_shape),
    ):
        _validate_tensor(
            tensor, name, shape=expected, dtype=torch.bfloat16, device=device
        )
    _validate_tensor(
        beta,
        "beta",
        shape=(batch, tokens, num_v_heads),
        dtype=torch.bfloat16,
        device=device,
    )
    _validate_tensor(
        A_log, "A_log", shape=(num_v_heads,), dtype=torch.float32, device=device
    )
    _validate_tensor(
        dt_bias,
        "dt_bias",
        shape=(num_v_heads, _HEAD_DIM),
        dtype=torch.float32,
        device=device,
    )
    num_sequences = len(seq_lens)
    _validate_tensor(
        initial_state,
        "initial_state",
        shape=(num_sequences, num_v_heads, _HEAD_DIM, _HEAD_DIM),
        dtype=torch.float32,
        device=device,
    )
    return _TrainingShape(
        layout=layout,
        public_batch=batch,
        public_tokens=tokens,
        total_tokens=batch * tokens,
        num_sequences=num_sequences,
        num_qk_heads=num_qk_heads,
        num_v_heads=num_v_heads,
        seq_lens=seq_lens,
        offsets=offsets,
        route=_select_training_route(seq_lens, num_qk_heads, num_v_heads),
    )


def _training_target(device: torch.device) -> _TrainingTarget:
    return "sm100a" if get_compute_capability(device) == (10, 0) else "sm103a"


def _load_training_module(device: torch.device):
    from .jit.flash_kda_training import load_flash_kda_training_module

    return load_flash_kda_training_module(_training_target(device))


def _get_training_module(device: torch.device):
    # Keep the historical monkeypatch seam on flashinfer.kda_training.
    from . import kda_training

    return kda_training._get_training_module(device)


def _aligned_u8(device: torch.device, size: int) -> torch.Tensor:
    raw = torch.empty(size + 63, dtype=torch.uint8, device=device)
    offset = (-raw.data_ptr()) % 64
    return raw[offset : offset + size]


def _canonical_views(
    shape: _TrainingShape,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    return (
        q.view(1, shape.total_tokens, shape.num_qk_heads, _HEAD_DIM),
        k.view(1, shape.total_tokens, shape.num_qk_heads, _HEAD_DIM),
        v.view(1, shape.total_tokens, shape.num_v_heads, _HEAD_DIM),
        g.view(1, shape.total_tokens, shape.num_v_heads, _HEAD_DIM),
        beta.view(1, shape.total_tokens, shape.num_v_heads),
    )


def _final_grid_ctas(shape: _TrainingShape) -> int:
    total_tiles = shape.num_sequences * shape.num_v_heads
    if total_tiles <= _FINAL_SHORT_GRID_CTAS:
        return total_tiles
    max_chunks = max((length + 63) // 64 for length in shape.seq_lens)
    if max_chunks <= 8:
        return min(_FINAL_SHORT_GRID_CTAS, total_tiles)
    return min(_FINAL_LONG_GRID_CTAS, total_tiles)


def _build_c16_metadata(
    shape: _TrainingShape, device: torch.device
) -> dict[str, object]:
    chunk_counts = tuple(length // _C16_CHUNK for length in shape.seq_lens)
    resident_ctas = int(torch.cuda.get_device_properties(device).multi_processor_count)
    token_starts = [0]
    checkpoint_starts = [0]
    for length, count in zip(shape.seq_lens, chunk_counts, strict=True):
        token_starts.append(token_starts[-1] + length)
        checkpoint_starts.append(checkpoint_starts[-1] + count)
    split = shape.num_v_heads <= 8 and max(shape.seq_lens) >= 1024
    rows: list[tuple[int, int, int, int, int, int, int, int, int]] = []
    n_tiles = shape.num_sequences * shape.num_v_heads
    ideal_tokens = (
        shape.total_tokens * shape.num_v_heads + resident_ctas - 1
    ) // resident_ctas
    ideal_chunks = max(1, (ideal_tokens + _C16_CHUNK - 1) // _C16_CHUNK)
    for sequence, count in enumerate(chunk_counts):
        piece_count = 1
        if split:
            piece_cap = max(1, min(count, 2048))
            piece_count = max(
                1, min((count + ideal_chunks - 1) // ideal_chunks, piece_cap)
            )
            if n_tiles < 2 * resident_ctas:
                piece_start = max(1, piece_count - 8)
                best_cost = 2**31 - 1
                for delta in range(16):
                    candidate = min(piece_start + delta, piece_cap)
                    span = (count + candidate - 1) // candidate
                    waves = (n_tiles * candidate + resident_ctas - 1) // resident_ctas
                    cost = waves * (span + 16)
                    if cost < best_cost:
                        best_cost, piece_count = cost, candidate
                unsplit_cost = ((n_tiles + resident_ctas - 1) // resident_ctas) * (
                    count + 16
                )
                if 4 * best_cost > 3 * unsplit_cost:
                    piece_count = 1
        span = (count + piece_count - 1) // piece_count
        num_pieces = (count + span - 1) // span
        for head in range(shape.num_v_heads):
            for piece in range(num_pieces):
                write_start = piece * span
                write_end = min(count, write_start + span)
                rows.append(
                    (
                        sequence,
                        head,
                        piece,
                        write_start,
                        write_end,
                        0 if piece == 0 else write_start,
                        count if piece + 1 == num_pieces else write_end,
                        token_starts[sequence],
                        token_starts[sequence + 1],
                    )
                )
    rows.sort(key=lambda row: row[4] - row[3], reverse=True)
    row_index = {(row[0], row[1], row[2]): index for index, row in enumerate(rows)}
    boundaries: list[tuple[int, int]] = []
    for sequence in range(shape.num_sequences):
        pieces = max(row[2] for row in rows if row[0] == sequence) + 1
        for head in range(shape.num_v_heads):
            for piece in range(1, pieces):
                boundaries.append(
                    (
                        row_index[(sequence, head, piece - 1)],
                        row_index[(sequence, head, piece)],
                    )
                )
    work_rows = [
        (sequence, head, write_start, write_end, compute_start, compute_end, bos, eos)
        for sequence, head, _piece, write_start, write_end, compute_start, compute_end, bos, eos in rows
    ]
    base_work_items = torch.tensor(work_rows, dtype=torch.int32, device=device)
    return {
        "chunk_counts": chunk_counts,
        "total_chunks": sum(chunk_counts),
        "base_work_items": base_work_items,
        "work_items": base_work_items.clone(),
        "boundaries": torch.tensor(
            boundaries if boundaries else [(0, 0)], dtype=torch.int32, device=device
        ),
        "checkpoint_cu_starts": torch.tensor(
            checkpoint_starts, dtype=torch.int64, device=device
        ),
        "total_work_items": len(rows),
        "uniform_work_items": int(not boundaries and len(set(chunk_counts)) == 1),
        "boundary_count": len(boundaries),
    }


def _build_c32_metadata(
    shape: _TrainingShape, device: torch.device
) -> dict[str, object]:
    sm_count = int(torch.cuda.get_device_properties(device).multi_processor_count)
    total_sequence_heads = shape.num_sequences * shape.num_v_heads
    chunk_offsets = [0]
    chunk_counts: list[int] = []
    chunk_sequences: list[int] = []
    chunk_indices: list[int] = []
    for sequence, length in enumerate(shape.seq_lens):
        count = (length + _C32_CHUNK - 1) // _C32_CHUNK
        chunk_counts.append(count)
        chunk_offsets.append(chunk_offsets[-1] + count)
        chunk_sequences.extend([sequence] * count)
        chunk_indices.extend(range(count))
    sequence_order = sorted(
        range(shape.num_sequences),
        key=lambda index: shape.seq_lens[index],
        reverse=True,
    )
    split_multiplier = sm_count // total_sequence_heads
    forward_two_wave_fill = (
        sm_count // 2 < total_sequence_heads < sm_count and max(chunk_counts) >= 96
    )
    use_split = (
        split_multiplier >= 2 and max(chunk_counts) >= 64
    ) or forward_two_wave_fill
    rows: list[tuple[int, int, int, int, int]] = []
    for sequence in sequence_order:
        count = chunk_counts[sequence]
        splits = 1
        if use_split:
            split_cap = (
                (2 * sm_count) // total_sequence_heads
                if forward_two_wave_fill
                else split_multiplier
            )
            splits = min(split_cap, max(1, count // 32))
        for head in range(shape.num_v_heads):
            for split in range(splits):
                rows.append(
                    (
                        sequence,
                        head,
                        (count * split) // splits,
                        (count * (split + 1)) // splits,
                        count,
                    )
                )
    rows.sort(key=lambda row: row[3] - row[2], reverse=True)
    boundary_split_items = use_split and not forward_two_wave_fill
    split_boundary = shape.num_v_heads <= 64 and not boundary_split_items
    boundary_value_splits = 2 if split_boundary else 1
    boundary_multiplier = max(
        1, sm_count // (total_sequence_heads * boundary_value_splits)
    )
    boundary_rows: list[tuple[int, int, int, int, int]] = []
    for sequence in sequence_order:
        count = chunk_counts[sequence]
        splits = (
            min(boundary_multiplier, max(1, count // 32)) if boundary_split_items else 1
        )
        for head in range(shape.num_v_heads):
            for split in range(splits):
                boundary_rows.append(
                    (
                        sequence,
                        head,
                        (count * split) // splits,
                        (count * (split + 1)) // splits,
                        count,
                    )
                )
    boundary_rows.sort(key=lambda row: row[3] - row[2], reverse=True)
    consumer_chunks = [
        chunk_offsets[sequence] + chunk_counts[sequence] - 1 - reverse_depth
        for reverse_depth in range(max(chunk_counts))
        for sequence in sequence_order
        if reverse_depth < chunk_counts[sequence]
    ]
    pair_starts = [
        chunk_offsets[sequence] + local
        for sequence, count in enumerate(chunk_counts)
        for local in range(0, count, 2)
    ]
    return {
        "chunk_counts": tuple(chunk_counts),
        "total_chunks": chunk_offsets[-1],
        "cu_chunk_offsets": torch.tensor(
            chunk_offsets, dtype=torch.int64, device=device
        ),
        "chunk_sequence": torch.tensor(
            chunk_sequences, dtype=torch.int32, device=device
        ),
        "chunk_index": torch.tensor(chunk_indices, dtype=torch.int32, device=device),
        "seq_order": torch.tensor(sequence_order, dtype=torch.int32, device=device),
        "work_items": torch.tensor(rows, dtype=torch.int32, device=device),
        "boundary_work_items": torch.tensor(
            boundary_rows, dtype=torch.int32, device=device
        ),
        "consumer_chunk_order": torch.tensor(
            consumer_chunks, dtype=torch.int32, device=device
        ),
        "chunk_pair_start": torch.tensor(pair_starts, dtype=torch.int32, device=device),
        "num_work_items": len(rows),
        "boundary_count": len(boundary_rows),
        "split_boundary": split_boundary,
        "boundary_value_splits": boundary_value_splits,
        "total_pairs": len(pair_starts),
        "use_split_work_items": use_split,
        "forward_two_wave_fill": forward_two_wave_fill,
        "boundary_split_work_items": boundary_split_items,
    }


@dataclass
class RecurrentKDATrainingContext:
    """Route-tagged forward tapes consumed by the paired backward."""

    state_checkpoints: torch.Tensor
    beta_active: torch.Tensor
    _route: _TrainingRouteSpec = field(repr=False)
    _shape: _TrainingShape = field(repr=False)
    _q: torch.Tensor = field(repr=False)
    _k: torch.Tensor = field(repr=False)
    _v: torch.Tensor = field(repr=False)
    _g: torch.Tensor = field(repr=False)
    _beta: torch.Tensor = field(repr=False)
    _A_log: torch.Tensor = field(repr=False)
    _dt_bias: torch.Tensor = field(repr=False)
    _initial_state: torch.Tensor = field(repr=False)
    _cu_seqlens: torch.Tensor = field(repr=False)
    _route_tensors: dict[str, torch.Tensor] = field(repr=False)
    _metadata: dict[str, object] = field(repr=False)
    _final_output_scratch: torch.Tensor = field(repr=False)
    _final_descriptor_storage: torch.Tensor = field(repr=False)
    _final_tensormap_workspace: torch.Tensor = field(repr=False)
    _dummy_f32: torch.Tensor = field(repr=False)
    _dummy_i32: torch.Tensor = field(repr=False)
    _final_grid_ctas: int = field(repr=False)
    _stream_ptr: int = field(repr=False)
    _input_tensors: tuple[torch.Tensor, ...] = field(default=(), repr=False)
    _input_signatures: tuple[tuple, ...] = field(default=(), repr=False)
    _saved_context_signatures: tuple[tuple, ...] = field(default=(), repr=False)
    _final_descriptor_signature: Optional[tuple] = field(default=None, repr=False)
    _route_descriptor_signature: Optional[tuple] = field(default=None, repr=False)
    _backward_buffers: dict[str, torch.Tensor] = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


def _saved_context_tensors(
    context: RecurrentKDATrainingContext,
) -> tuple[torch.Tensor, ...]:
    metadata_tensors = tuple(
        value for value in context._metadata.values() if isinstance(value, torch.Tensor)
    )
    return (
        context.state_checkpoints,
        context.beta_active,
        context._cu_seqlens,
        *context._route_tensors.values(),
        *metadata_tensors,
    )


def _new_context(
    shape: _TrainingShape,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor],
    stream_ptr: int,
) -> RecurrentKDATrainingContext:
    device = q.device
    qf, kf, vf, _gf, betaf = _canonical_views(shape, q, k, v, g, beta)
    canonical_cu = (
        torch.tensor(shape.offsets, dtype=torch.int64, device=device)
        if cu_seqlens is None
        else cu_seqlens
    )
    route_tensors: dict[str, torch.Tensor] = {}
    if shape.route.family == "c16":
        metadata = _build_c16_metadata(shape, device)
        total_chunks = cast(int, metadata["total_chunks"])
        state_checkpoints = torch.empty(
            (total_chunks, shape.num_v_heads, _HEAD_DIM, _HEAD_DIM),
            dtype=torch.bfloat16,
            device=device,
        )
        beta_active = torch.empty(
            (shape.total_tokens, max(shape.num_v_heads, 8)),
            dtype=torch.bfloat16,
            device=device,
        )
        route_tensors["counters"] = torch.empty(
            shape.num_v_heads + 2, dtype=torch.uint32, device=device
        )
    elif shape.route.family == "row_split":
        metadata = {"total_chunks": shape.total_tokens}
        state_checkpoints = torch.empty(
            (shape.total_tokens, shape.num_v_heads, _HEAD_DIM, _HEAD_DIM),
            dtype=torch.float32,
            device=device,
        )
        beta_active = torch.empty(
            (shape.total_tokens, shape.num_v_heads), dtype=torch.float32, device=device
        )
        vector_shape = (shape.total_tokens, shape.num_v_heads, _HEAD_DIM)
        route_tensors.update(
            q_norm=torch.empty(vector_shape, dtype=torch.float32, device=device),
            k_norm=torch.empty(vector_shape, dtype=torch.float32, device=device),
            decay=torch.empty(vector_shape, dtype=torch.float32, device=device),
        )
    else:
        metadata = _build_c32_metadata(shape, device)
        total_chunks = cast(int, metadata["total_chunks"])
        token_vector = (shape.total_tokens, shape.num_v_heads, _HEAD_DIM)
        tape_vector = (total_chunks, shape.num_v_heads, _C32_CHUNK, _HEAD_DIM)
        tape_value = (total_chunks, shape.num_v_heads, _HEAD_DIM, _C32_CHUNK)
        grouped = shape.route.grouped
        route_tensors["q_value_heads"] = (
            torch.empty(
                (1, shape.total_tokens, shape.num_v_heads, _HEAD_DIM),
                dtype=torch.bfloat16,
                device=device,
            )
            if grouped
            else qf
        )
        route_tensors["k_value_heads"] = (
            torch.empty_like(route_tensors["q_value_heads"]) if grouped else kf
        )
        padded_beta_heads = ((shape.num_v_heads + 7) // 8) * 8
        route_tensors["beta_tma"] = (
            betaf.view(shape.total_tokens, shape.num_v_heads)
            if shape.total_tokens >= _C32_CHUNK
            and padded_beta_heads == shape.num_v_heads
            else torch.empty(
                (max(shape.total_tokens, _C32_CHUNK), padded_beta_heads),
                dtype=torch.bfloat16,
                device=device,
            )
        )
        state_checkpoints = torch.empty(
            (total_chunks, shape.num_v_heads, _HEAD_DIM, _HEAD_DIM),
            dtype=torch.bfloat16,
            device=device,
        )
        beta_active = torch.empty(
            (shape.total_tokens, shape.num_v_heads), dtype=torch.float32, device=device
        )
        state_votes = total_chunks * shape.num_v_heads + max(
            shape.num_sequences * shape.num_v_heads,
            cast(int, metadata["num_work_items"]),
        )
        route_tensors.update(
            state_checkpoint_needed=torch.empty(
                state_votes, dtype=torch.uint32, device=device
            ),
            tape_qd=torch.empty(tape_vector, dtype=torch.bfloat16, device=device),
            tape_kd=torch.empty(tape_vector, dtype=torch.bfloat16, device=device),
            tape_kr=torch.empty(tape_vector, dtype=torch.bfloat16, device=device),
            tape_j=torch.empty(
                (total_chunks, shape.num_v_heads, _C32_CHUNK, _C32_CHUNK),
                dtype=torch.bfloat16,
                device=device,
            ),
            tape_restore_factor=torch.empty(
                (total_chunks, shape.num_v_heads, _HEAD_DIM),
                dtype=torch.float32,
                device=device,
            ),
            tape_x=torch.empty(tape_value, dtype=torch.bfloat16, device=device),
            tape_r=torch.empty(tape_value, dtype=torch.bfloat16, device=device),
            norm_inv=torch.empty(
                (shape.total_tokens, shape.num_v_heads, 2),
                dtype=torch.float32,
                device=device,
            ),
            decay=torch.empty(token_vector, dtype=torch.bfloat16, device=device),
            zero_workspace=torch.empty(
                total_chunks * shape.num_v_heads, dtype=torch.uint32, device=device
            ),
            descriptor_storage=_aligned_u8(device, 7 * 128),
        )
        route_tensors["tape_e"] = route_tensors["tape_x"]
    final_grid = _final_grid_ctas(shape)
    return RecurrentKDATrainingContext(
        state_checkpoints=state_checkpoints,
        beta_active=beta_active,
        _route=shape.route,
        _shape=shape,
        _q=q,
        _k=k,
        _v=v,
        _g=g,
        _beta=beta,
        _A_log=A_log,
        _dt_bias=dt_bias,
        _initial_state=initial_state,
        _cu_seqlens=canonical_cu,
        _route_tensors=route_tensors,
        _metadata=metadata,
        _final_output_scratch=torch.empty_like(vf),
        _final_descriptor_storage=_aligned_u8(device, 5 * 128),
        _final_tensormap_workspace=torch.empty(
            final_grid * _FINAL_TMAP_SLOTS_PER_CTA * _FINAL_TMAP_BYTES_PER_SLOT,
            dtype=torch.uint8,
            device=device,
        ),
        _dummy_f32=torch.empty(1, dtype=torch.float32, device=device),
        _dummy_i32=torch.empty(1, dtype=torch.int32, device=device),
        _final_grid_ctas=final_grid,
        _stream_ptr=stream_ptr,
    )


def _validate_context_storage(
    context: RecurrentKDATrainingContext, device: torch.device
) -> None:
    if context._shape.route != context._route:
        raise ValueError("context route tag does not match its normalized shape")
    named = [
        ("state_checkpoints", context.state_checkpoints),
        ("beta_active", context.beta_active),
        *context._route_tensors.items(),
        *(
            (name, value)
            for name, value in context._metadata.items()
            if isinstance(value, torch.Tensor)
        ),
    ]
    for name, tensor in named:
        if not tensor.is_cuda or tensor.device != device or not tensor.is_contiguous():
            raise ValueError(f"context tensor {name} must be contiguous on {device}")
    if context._route.family == "c32":
        if (
            context._route_tensors["tape_e"].data_ptr()
            != context._route_tensors["tape_x"].data_ptr()
        ):
            raise ValueError("C32 tape_e must exactly alias tape_x")
        if context._route_tensors["descriptor_storage"].data_ptr() % 64:
            raise ValueError("C32 descriptor storage must be 64-byte aligned")
    if context._final_descriptor_storage.data_ptr() % 64:
        raise ValueError("final descriptor storage must be 64-byte aligned")


def _descriptor_signature(*tensors: torch.Tensor) -> tuple:
    return tuple(_tensor_signature(tensor) for tensor in tensors)


def _run_forward_route(
    context: RecurrentKDATrainingContext,
    output_flat: torch.Tensor,
    final_state: torch.Tensor,
    scale: float,
    lower_bound: float,
) -> None:
    shape = context._shape
    qf, kf, vf, gf, betaf = _canonical_views(
        shape, context._q, context._k, context._v, context._g, context._beta
    )
    module = _get_training_module(context._q.device)
    route = context._route.family
    final_target = context._final_output_scratch if route == "c16" else output_flat
    final_signature = _descriptor_signature(
        qf, kf, vf, gf, final_target, context._final_descriptor_storage
    )
    prepare_final = int(final_signature != context._final_descriptor_signature)
    if route == "c16":
        m = context._metadata
        t = context._route_tensors
        module.run_training_forward(
            qf,
            kf,
            vf,
            gf,
            betaf,
            context._A_log,
            context._dt_bias,
            context._initial_state,
            context._cu_seqlens,
            m["checkpoint_cu_starts"],
            m["base_work_items"],
            m["work_items"],
            m["boundaries"],
            t["counters"],
            output_flat,
            final_state,
            context.state_checkpoints,
            context.beta_active,
            context._final_output_scratch,
            context._final_descriptor_storage,
            context._final_tensormap_workspace,
            context._dummy_f32,
            context._dummy_i32,
            m["boundary_count"],
            m["total_work_items"],
            shape.total_tokens,
            shape.num_sequences,
            shape.num_qk_heads,
            shape.num_v_heads,
            m["total_chunks"],
            max(shape.num_v_heads, 8),
            m["uniform_work_items"],
            context._final_grid_ctas,
            prepare_final,
            scale,
            lower_bound,
            context._stream_ptr,
        )
    elif route == "row_split":
        t = context._route_tensors
        module.run_training_row_forward(
            qf,
            kf,
            vf,
            gf,
            betaf,
            context._A_log,
            context._dt_bias,
            context._initial_state,
            context._cu_seqlens,
            output_flat,
            final_state,
            t["q_norm"],
            t["k_norm"],
            t["decay"],
            context.beta_active,
            context.state_checkpoints,
            context._final_descriptor_storage,
            context._final_tensormap_workspace,
            context._dummy_f32,
            context._dummy_i32,
            shape.total_tokens,
            shape.num_sequences,
            shape.num_v_heads,
            context._final_grid_ctas,
            prepare_final,
            scale,
            lower_bound,
            context._stream_ptr,
        )
    else:
        t = context._route_tensors
        m = context._metadata
        route_signature = _descriptor_signature(
            t["q_value_heads"],
            t["k_value_heads"],
            vf,
            gf,
            t["beta_tma"],
            output_flat,
            context.state_checkpoints,
            t["descriptor_storage"],
        )
        prepare_route = int(route_signature != context._route_descriptor_signature)
        module.run_training_c32_forward(
            qf,
            kf,
            t["q_value_heads"],
            t["k_value_heads"],
            vf,
            gf,
            betaf,
            t["beta_tma"],
            context._A_log,
            context._dt_bias,
            context._initial_state,
            context._cu_seqlens,
            m["work_items"],
            m["seq_order"],
            m["cu_chunk_offsets"],
            t["descriptor_storage"],
            output_flat,
            final_state,
            context.state_checkpoints,
            t["state_checkpoint_needed"],
            t["tape_qd"],
            t["tape_kd"],
            t["tape_kr"],
            t["tape_j"],
            t["tape_restore_factor"],
            t["tape_e"],
            t["tape_x"],
            t["tape_r"],
            t["norm_inv"],
            t["decay"],
            context.beta_active,
            t["zero_workspace"],
            final_target,
            context._final_descriptor_storage,
            context._final_tensormap_workspace,
            context._dummy_f32,
            context._dummy_i32,
            shape.total_tokens,
            shape.num_sequences,
            shape.num_qk_heads,
            shape.num_v_heads,
            m["total_chunks"],
            m["num_work_items"],
            int(cast(bool, m["use_split_work_items"])),
            int(context._route.grouped),
            prepare_route,
            prepare_final,
            context._final_grid_ctas,
            scale,
            lower_bound,
            context._stream_ptr,
        )
        context._route_descriptor_signature = route_signature
    context._final_descriptor_signature = final_signature


def _recurrent_kda_training_forward_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    lower_bound: float = _LOWER_BOUND,
    out: Optional[torch.Tensor] = None,
    final_state_out: Optional[torch.Tensor] = None,
    context_out: Optional[RecurrentKDATrainingContext] = None,
) -> tuple[torch.Tensor, torch.Tensor, RecurrentKDATrainingContext]:
    shape = _validate_forward_inputs(
        q, k, v, g, beta, A_log, dt_bias, initial_state, cu_seqlens
    )
    scale_value, lower_bound_value = _validate_scale_and_bound(scale, lower_bound)
    device = q.device
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "recurrent KDA training does not support CUDA graph capture"
            )
    stream_ptr = int(torch.cuda.current_stream(device).cuda_stream)
    output = torch.empty_like(v) if out is None else out
    final_state = (
        torch.empty_like(initial_state) if final_state_out is None else final_state_out
    )
    _validate_tensor(
        output, "out", shape=tuple(v.shape), dtype=torch.bfloat16, device=device
    )
    _validate_tensor(
        final_state,
        "final_state_out",
        shape=tuple(initial_state.shape),
        dtype=torch.float32,
        device=device,
    )
    public_inputs: tuple[torch.Tensor, ...] = (
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        initial_state,
    )
    input_names: tuple[str, ...] = (
        "q",
        "k",
        "v",
        "g",
        "beta",
        "A_log",
        "dt_bias",
        "initial_state",
    )
    if cu_seqlens is not None:
        public_inputs += (cu_seqlens,)
        input_names += ("cu_seqlens",)
    _check_writes_do_not_overlap(
        (("out", output), ("final_state_out", final_state)),
        tuple(zip(input_names, public_inputs, strict=True)),
    )
    if context_out is None:
        context = _new_context(
            shape,
            q,
            k,
            v,
            g,
            beta,
            A_log,
            dt_bias,
            initial_state,
            cu_seqlens,
            stream_ptr,
        )
    else:
        context = context_out
        if context._stream_ptr != stream_ptr:
            raise RuntimeError(
                "a recurrent KDA training context must be reused on its forward stream"
            )
        if context._shape != shape or context._route != shape.route:
            raise ValueError(
                "context_out layout, shape, or route does not match the new inputs"
            )
        _validate_context_storage(context, device)
        if (
            context._saved_context_signatures
            and tuple(
                _tensor_signature(tensor) for tensor in _saved_context_tensors(context)
            )
            != context._saved_context_signatures
        ):
            raise RuntimeError(
                "the recurrent KDA training context was modified after forward"
            )
        if cu_seqlens is not None:
            context._cu_seqlens = cu_seqlens
    _qf, _kf, vf, _gf, _betaf = _canonical_views(shape, q, k, v, g, beta)
    context._q, context._k, context._v, context._g = q, k, v, g
    context._beta, context._A_log, context._dt_bias = beta, A_log, dt_bias
    context._initial_state = initial_state
    internal_writes: list[tuple[str, torch.Tensor]] = [
        ("context.state_checkpoints", context.state_checkpoints),
        ("context.beta_active", context.beta_active),
        ("context._final_output_scratch", context._final_output_scratch),
        ("context._final_descriptor_storage", context._final_descriptor_storage),
        ("context._final_tensormap_workspace", context._final_tensormap_workspace),
    ]
    if context._route.family == "c16":
        internal_writes.extend(
            (
                ("context._work_items", context._metadata["work_items"]),
                ("context._counters", context._route_tensors["counters"]),
            )
        )
    elif context._route.family == "row_split":
        internal_writes.extend(
            (f"context.{name}", context._route_tensors[name])
            for name in ("q_norm", "k_norm", "decay")
        )
    else:
        c32_write_names = [
            "state_checkpoint_needed",
            "tape_qd",
            "tape_kd",
            "tape_kr",
            "tape_j",
            "tape_restore_factor",
            "tape_x",
            "tape_r",
            "norm_inv",
            "decay",
            "zero_workspace",
            "descriptor_storage",
        ]
        if context._route.grouped:
            c32_write_names.extend(("q_value_heads", "k_value_heads"))
        beta_tma = context._route_tensors["beta_tma"]
        if not _storage_ranges_overlap(beta_tma, beta):
            c32_write_names.append("beta_tma")
        internal_writes.extend(
            (f"context.{name}", context._route_tensors[name])
            for name in c32_write_names
        )
    _check_writes_do_not_overlap(
        (("out", output), ("final_state_out", final_state), *internal_writes),
        tuple(zip(input_names, public_inputs, strict=True)),
    )
    _run_forward_route(
        context, output.view_as(vf), final_state, scale_value, lower_bound_value
    )
    context._input_tensors = public_inputs
    context._input_signatures = tuple(
        _tensor_signature(tensor) for tensor in public_inputs
    )
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
    cu_seqlens: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    lower_bound: float = _LOWER_BOUND,
    out: Optional[torch.Tensor] = None,
    final_state_out: Optional[torch.Tensor] = None,
    context_out: Optional[RecurrentKDATrainingContext] = None,
) -> tuple[torch.Tensor, torch.Tensor, RecurrentKDATrainingContext]:
    r"""Run fixed or packed KDA forward and save the selected route's tapes."""

    args = (
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
    )
    if context_out is None:
        return _recurrent_kda_training_forward_impl(*args, None)
    if not isinstance(context_out, RecurrentKDATrainingContext):
        raise TypeError("context_out must be a RecurrentKDATrainingContext")
    with context_out._lock:
        return _recurrent_kda_training_forward_impl(*args, context_out)


def _gradient_outputs(
    out: Optional[Sequence[torch.Tensor]], context: RecurrentKDATrainingContext
) -> tuple[torch.Tensor, ...]:
    expected = (
        (context._q.shape, torch.bfloat16),
        (context._k.shape, torch.bfloat16),
        (context._v.shape, torch.bfloat16),
        (context._g.shape, torch.bfloat16),
        (context._beta.shape, torch.bfloat16),
        (context._A_log.shape, torch.float32),
        (context._dt_bias.shape, torch.float32),
        (context._initial_state.shape, torch.float32),
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
            tensor,
            name,
            shape=tuple(shape),
            dtype=dtype,
            device=context._q.device,
        )
    return tuple(out)


def _backward_buffer(
    context: RecurrentKDATrainingContext,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    tensor = context._backward_buffers.get(name)
    if tensor is None:
        tensor = torch.empty(shape, dtype=dtype, device=context._q.device)
        context._backward_buffers[name] = tensor
    else:
        _validate_tensor(
            tensor,
            f"context._backward_buffers[{name!r}]",
            shape=shape,
            dtype=dtype,
            device=context._q.device,
        )
    return tensor


def _run_c16_backward(
    context: RecurrentKDATrainingContext,
    do_flat: torch.Tensor,
    dfinal_state: torch.Tensor,
    outputs: tuple[torch.Tensor, ...],
) -> None:
    shape, m, t = context._shape, context._metadata, context._route_tensors
    qf, kf, vf, gf, _ = _canonical_views(
        shape, context._q, context._k, context._v, context._g, context._beta
    )
    dq, dk, dv, dg, dbeta, dA, ddt, dinit = outputs
    dqf, dkf = dq.view_as(qf), dk.view_as(kf)
    dvf, dgf = dv.view_as(vf), dg.view_as(gf)
    dbetaf = dbeta.view(1, shape.total_tokens, shape.num_v_heads)
    vector = (shape.total_tokens, shape.num_v_heads, _HEAD_DIM)
    dlog = _backward_buffer(context, "c16_dlog_decay", vector, torch.float32)
    dboundary = _backward_buffer(
        context,
        "c16_dlog_boundary",
        (cast(int, m["total_chunks"]), shape.num_v_heads, _HEAD_DIM),
        torch.float32,
    )
    dbeta_active = _backward_buffer(
        context,
        "c16_dbeta_active",
        (shape.total_tokens, shape.num_v_heads),
        torch.float32,
    )
    gate_a = _backward_buffer(
        context, "c16_gate_a", (128, shape.num_v_heads, _HEAD_DIM), torch.float32
    )
    gate_dt = _backward_buffer(
        context, "c16_gate_dt", (128, shape.num_v_heads, _HEAD_DIM), torch.float32
    )
    dummy_u32 = _backward_buffer(context, "c16_dummy_u32", (1,), torch.uint32)
    dummy_f32 = _backward_buffer(context, "c16_dummy_f32", (1,), torch.float32)
    grouped = context._route.grouped
    dq_value = (
        _backward_buffer(context, "c16_dq_value", tuple(vf.shape), torch.bfloat16)
        if grouped
        else dqf
    )
    dk_value = (
        _backward_buffer(context, "c16_dk_value", tuple(vf.shape), torch.bfloat16)
        if grouped
        else dkf
    )
    _get_training_module(context._q.device).run_training_backward(
        qf,
        kf,
        vf,
        gf,
        context._A_log,
        context._dt_bias,
        do_flat,
        dfinal_state,
        context._cu_seqlens,
        m["checkpoint_cu_starts"],
        m["work_items"],
        t["counters"],
        context.state_checkpoints,
        context.beta_active,
        dlog,
        dboundary,
        dbeta_active,
        gate_a,
        gate_dt,
        dummy_u32,
        dummy_f32,
        dq_value,
        dk_value,
        dvf,
        dgf,
        dbetaf,
        dA,
        ddt,
        dinit,
        dqf,
        dkf,
        m["total_work_items"],
        shape.total_tokens,
        shape.num_sequences,
        shape.num_qk_heads,
        shape.num_v_heads,
        m["total_chunks"],
        max(shape.num_v_heads, 8),
        m["uniform_work_items"],
        int(grouped),
        _SCALE,
        _LOWER_BOUND,
        context._stream_ptr,
    )


def _run_row_backward(
    context: RecurrentKDATrainingContext,
    do_flat: torch.Tensor,
    dfinal_state: torch.Tensor,
    outputs: tuple[torch.Tensor, ...],
) -> None:
    shape, t = context._shape, context._route_tensors
    qf, kf, vf, gf, _ = _canonical_views(
        shape, context._q, context._k, context._v, context._g, context._beta
    )
    dq, dk, dv, dg, dbeta, dA, ddt, dinit = outputs
    vector = (shape.total_tokens, shape.num_v_heads, _HEAD_DIM)
    dq_norm = _backward_buffer(context, "row_dq_norm", vector, torch.float32)
    dk_norm = _backward_buffer(context, "row_dk_norm", vector, torch.float32)
    dlog = _backward_buffer(context, "row_dlog", vector, torch.float32)
    dbeta_active = _backward_buffer(
        context,
        "row_dbeta_active",
        (shape.total_tokens, shape.num_v_heads),
        torch.float32,
    )
    _get_training_module(context._q.device).run_training_row_backward(
        qf,
        kf,
        vf,
        gf,
        context._A_log,
        context._dt_bias,
        context._initial_state,
        do_flat,
        dfinal_state,
        context._cu_seqlens,
        t["q_norm"],
        t["k_norm"],
        t["decay"],
        context.beta_active,
        context.state_checkpoints,
        dq_norm,
        dk_norm,
        dlog,
        dbeta_active,
        dq.view_as(qf),
        dk.view_as(kf),
        dv.view_as(vf),
        dg.view_as(gf),
        dbeta.view(1, shape.total_tokens, shape.num_v_heads),
        dA,
        ddt,
        dinit,
        shape.total_tokens,
        shape.num_sequences,
        shape.num_v_heads,
        _SCALE,
        _LOWER_BOUND,
        context._stream_ptr,
    )


def _run_c32_backward(
    context: RecurrentKDATrainingContext,
    do_flat: torch.Tensor,
    dfinal_state: torch.Tensor,
    outputs: tuple[torch.Tensor, ...],
) -> None:
    shape, t, m = context._shape, context._route_tensors, context._metadata
    qf, kf, vf, gf, _ = _canonical_views(
        shape, context._q, context._k, context._v, context._g, context._beta
    )
    dq, dk, dv, dg, dbeta, dA, ddt, dinit = outputs
    total_chunks = cast(int, m["total_chunks"])
    tape_value = (total_chunks, shape.num_v_heads, _HEAD_DIM, _C32_CHUNK)
    tape_vector = (total_chunks, shape.num_v_heads, _C32_CHUNK, _HEAD_DIM)
    chunk_state_shape = (total_chunks, shape.num_v_heads, _HEAD_DIM, _HEAD_DIM)
    chunk_dh = _backward_buffer(
        context, "c32_chunk_dh", chunk_state_shape, torch.bfloat16
    )
    chunk_dr = _backward_buffer(context, "c32_chunk_dr", tape_value, torch.bfloat16)
    chunk_dx = _backward_buffer(context, "c32_chunk_dx", tape_value, torch.bfloat16)
    boundary_ready = _backward_buffer(
        context, "c32_boundary_ready", (total_chunks * shape.num_v_heads,), torch.uint32
    )
    grad_qd = _backward_buffer(context, "c32_grad_qd", tape_vector, torch.bfloat16)
    grad_kd = _backward_buffer(context, "c32_grad_kd", tape_vector, torch.bfloat16)
    grad_ki = _backward_buffer(context, "c32_grad_ki", tape_vector, torch.bfloat16)
    dlog = _backward_buffer(
        context,
        "c32_dlog",
        (shape.total_tokens, shape.num_v_heads, _HEAD_DIM),
        torch.float32,
    )
    dbeta_active = _backward_buffer(
        context,
        "c32_dbeta_active",
        (shape.total_tokens, shape.num_v_heads),
        torch.float32,
    )
    grouped = context._route.grouped
    dq_value = (
        _backward_buffer(context, "c32_dq_value", tuple(vf.shape), torch.bfloat16)
        if grouped
        else dq.view_as(qf)
    )
    dk_value = (
        _backward_buffer(context, "c32_dk_value", tuple(vf.shape), torch.bfloat16)
        if grouped
        else dk.view_as(kf)
    )
    _get_training_module(context._q.device).run_training_c32_backward(
        t["q_value_heads"],
        t["k_value_heads"],
        vf,
        gf,
        context._A_log,
        context._dt_bias,
        context._initial_state,
        do_flat,
        dfinal_state,
        context._cu_seqlens,
        m["cu_chunk_offsets"],
        m["boundary_work_items"],
        m["consumer_chunk_order"],
        m["chunk_sequence"],
        m["chunk_index"],
        m["chunk_pair_start"],
        context.state_checkpoints,
        t["state_checkpoint_needed"],
        t["tape_qd"],
        t["tape_kd"],
        t["tape_kr"],
        t["tape_j"],
        t["tape_restore_factor"],
        t["tape_e"],
        t["tape_x"],
        t["tape_r"],
        t["norm_inv"],
        t["decay"],
        context.beta_active,
        chunk_dh,
        chunk_dr,
        chunk_dx,
        boundary_ready,
        grad_qd,
        grad_kd,
        grad_ki,
        dlog,
        dbeta_active,
        dq_value,
        dk_value,
        dv.view_as(vf),
        dg.view_as(gf),
        dbeta.view(1, shape.total_tokens, shape.num_v_heads),
        dA,
        ddt,
        dinit,
        dq.view_as(qf),
        dk.view_as(kf),
        shape.total_tokens,
        shape.num_sequences,
        shape.num_qk_heads,
        shape.num_v_heads,
        total_chunks,
        m["total_pairs"],
        m["boundary_count"],
        int(cast(bool, m["split_boundary"])),
        int(grouped),
        _SCALE,
        _LOWER_BOUND,
        context._stream_ptr,
    )


@flashinfer_api
def recurrent_kda_training_backward(
    context: RecurrentKDATrainingContext,
    do: torch.Tensor,
    dfinal_state: torch.Tensor,
    out: Optional[Sequence[torch.Tensor]] = None,
) -> tuple[torch.Tensor, ...]:
    r"""Differentiate a saved route context without rerunning forward recurrence."""

    # Route helpers invoke run_training_backward or the route-specific FFI symbol.

    if not isinstance(context, RecurrentKDATrainingContext):
        raise TypeError("context must be a RecurrentKDATrainingContext")
    with context._lock:
        device = context._q.device
        with torch.cuda.device(device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "recurrent KDA training does not support CUDA graph capture"
                )
        if int(torch.cuda.current_stream(device).cuda_stream) != context._stream_ptr:
            raise RuntimeError(
                "recurrent KDA training backward must run on the forward stream"
            )
        _validate_context_storage(context, device)
        if (
            tuple(_tensor_signature(tensor) for tensor in context._input_tensors)
            != context._input_signatures
        ):
            raise RuntimeError(
                "a recurrent KDA training input was modified after forward"
            )
        if (
            tuple(
                _tensor_signature(tensor) for tensor in _saved_context_tensors(context)
            )
            != context._saved_context_signatures
        ):
            raise RuntimeError(
                "the recurrent KDA training context was modified after forward"
            )
        _validate_tensor(
            do, "do", shape=tuple(context._v.shape), dtype=torch.bfloat16, device=device
        )
        _validate_tensor(
            dfinal_state,
            "dfinal_state",
            shape=tuple(context._initial_state.shape),
            dtype=torch.float32,
            device=device,
        )
        outputs = _gradient_outputs(out, context)
        _check_writes_do_not_overlap(
            tuple(
                (name, tensor)
                for name, tensor in zip(
                    KDA_BACKWARD_GRADIENT_NAMES, outputs, strict=True
                )
            ),
            (
                *(
                    (f"input[{index}]", tensor)
                    for index, tensor in enumerate(context._input_tensors)
                ),
                ("do", do),
                ("dfinal_state", dfinal_state),
                *(
                    (f"saved[{index}]", tensor)
                    for index, tensor in enumerate(_saved_context_tensors(context))
                ),
            ),
        )
        do_flat = do.view(
            1, context._shape.total_tokens, context._shape.num_v_heads, _HEAD_DIM
        )
        if context._route.family == "c16":
            _run_c16_backward(context, do_flat, dfinal_state, outputs)
        elif context._route.family == "row_split":
            _run_row_backward(context, do_flat, dfinal_state, outputs)
        else:
            _run_c32_backward(context, do_flat, dfinal_state, outputs)
        return outputs


__all__ = [
    "RecurrentKDATrainingContext",
    "recurrent_kda_training_backward",
    "recurrent_kda_training_forward",
]
