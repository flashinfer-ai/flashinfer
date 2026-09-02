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

Compute-capability 10.0/10.3 backend for Minimax Sparse Attention.
"""

from __future__ import annotations

import math
import os
import threading
from contextlib import nullcontext
from typing import TYPE_CHECKING, Optional, cast

import torch

from ..utils import get_compute_capability

if TYPE_CHECKING:
    from ..jit.blackwell_msa import BlackwellMSATarget, BlackwellMSAVariant

_BLOCK_SIZE = 128
_HEAD_DIM = 128
_TOPK_SELECT = 16
_ATTENTION_TOPK = 16
_SUPPORTED_ATTENTION_TOPK = {4, 8, 16, 32}
_SUPPORTED_COMPUTE_CAPABILITIES = {(10, 0), (10, 3)}
_M128_Q_TILE = 256
_M128_GQA8_Q_TILE = 32
_M128_GQA16_Q_TILE = 16
_M64_GQA16_Q_TILE = 8
_UNIFORM_FP8_EVEN_WAVE_GRID = 128

_LONG_PARTIAL_SEGMENT_COUNT = 4
_LONG_WORK_GROUP_BUCKETS = (128, 64, 32, 16, 8, 4, 2, 1)
_LONG_GROUP_BOUNDARIES = _LONG_WORK_GROUP_BUCKETS[:-1]
_LONG_QSPLIT_Q_MASK = 0x00FF_FFFF
_LONG_QSPLIT_SLOT_SHIFT = 24
_LONG_QSPLIT_SINGLE_SHIFT = 28

_MODE_DECODE_ONLY = 1
_SPLIT_ADAPTIVE = 0


class MSASparseAttentionWorkspace:
    """Caller-owned storage for SM100/SM103 MSA CUDA graph capture.

    Construct one workspace per captured sparse-attention invocation. Warm it
    by calling the operation eagerly with the exact tensors, options, and CUDA
    stream that will be captured, then synchronize that stream before capture.
    The workspace owns output and temporary tensors whose addresses must stay
    stable for graph replay.

    A workspace binds to its first stream. Once it participates in capture it
    cannot be passed through Python again; graph replay remains valid for the
    lifetime of the workspace.
    """

    def __init__(self, device: torch.device | str) -> None:
        normalized_device = torch.device(device)
        if normalized_device.type != "cuda":
            raise ValueError("MSASparseAttentionWorkspace requires a CUDA device")
        if normalized_device.index is None:
            normalized_device = torch.device("cuda", torch.cuda.current_device())
        self.device = normalized_device
        self._lock = threading.Lock()
        self._buffers: dict[str, torch.Tensor] = {}
        self._long_prefill_state: dict = {}
        self._reverse_prefill_states: dict[str, dict] = {}
        self._warmed_launches: set[tuple] = set()
        self._bound_stream_ptr: Optional[int] = None
        self._captured = False


_topk_warmed_devices: set[tuple[int, str]] = set()
_topk_warmed_devices_lock = threading.Lock()
_eager_decode_dummies: dict[tuple[int, int, torch.dtype], torch.Tensor] = {}
_eager_decode_dummies_lock = threading.Lock()
_implicit_long_prefill_states: dict[tuple, dict] = {}
_implicit_long_prefill_states_lock = threading.Lock()
_implicit_reverse_prefill_states: dict[str, dict] = {}
_implicit_reverse_prefill_states_lock = threading.Lock()


def is_blackwell_msa_device(device: torch.device | str) -> bool:
    """Return whether ``device`` is a supported SM100/SM103 MSA target."""

    normalized_device = torch.device(device)
    return (
        normalized_device.type == "cuda"
        and get_compute_capability(normalized_device) in _SUPPORTED_COMPUTE_CAPABILITIES
    )


def _cuda_version_at_least(version: str) -> bool:
    from ..jit.cpp_ext import is_cuda_version_at_least

    return is_cuda_version_at_least(version)


def _select_target(device: torch.device) -> "BlackwellMSATarget":
    compute_capability = get_compute_capability(device)
    if compute_capability not in _SUPPORTED_COMPUTE_CAPABILITIES:
        raise RuntimeError(
            "the SM100/SM103 MSA backend requires compute capability 10.0 or 10.3; "
            f"got {compute_capability[0]}.{compute_capability[1]}"
        )
    if compute_capability == (10, 3):
        if _cuda_version_at_least("12.9"):
            return "sm103a"
        raise RuntimeError("MSA on compute capability 10.3 requires CUDA 12.9 or newer")
    if _cuda_version_at_least("12.8"):
        return "sm100a"
    raise RuntimeError("MSA on compute capability 10.0 requires CUDA 12.8 or newer")


def _get_module(variant: "BlackwellMSAVariant", target: "BlackwellMSATarget"):
    from ..jit.blackwell_msa import get_blackwell_msa_module

    return get_blackwell_msa_module(variant, target)


def _stream_ptr(device: torch.device) -> int:
    return int(torch.cuda.current_stream(device).cuda_stream)


def _decode_tma_dummy(
    *,
    device: torch.device,
    stream_ptr: int,
    dtype: torch.dtype,
    workspace: Optional[MSASparseAttentionWorkspace],
) -> torch.Tensor:
    if workspace is not None:
        return _workspace_buffer(
            workspace,
            f"decode_tma_dummy_{dtype}",
            (128, 2, _HEAD_DIM),
            dtype=dtype,
            device=device,
        )
    device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    key = (device_index, stream_ptr, dtype)
    with _eager_decode_dummies_lock:
        tensor = _eager_decode_dummies.get(key)
        if tensor is None:
            tensor = torch.empty(
                (128, 2, _HEAD_DIM),
                dtype=dtype,
                device=device,
            )
            _eager_decode_dummies[key] = tensor
    return tensor


def _normalize_device(device: torch.device) -> torch.device:
    if device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def _bind_workspace(
    workspace: MSASparseAttentionWorkspace,
    *,
    device: torch.device,
    stream_ptr: int,
    capturing: bool,
) -> None:
    device = _normalize_device(device)
    if workspace.device != device:
        raise ValueError(
            f"MSASparseAttentionWorkspace is bound to {workspace.device}, "
            f"but MSA inputs are on {device}"
        )
    if workspace._bound_stream_ptr is None:
        workspace._bound_stream_ptr = stream_ptr
    elif workspace._bound_stream_ptr != stream_ptr:
        raise RuntimeError(
            "MSASparseAttentionWorkspace is bound to a different CUDA stream; "
            "warm and capture it on one stream"
        )
    if workspace._captured:
        reuse_kind = "captured again" if capturing else "reused eagerly"
        raise RuntimeError(
            "MSASparseAttentionWorkspace has already participated in CUDA graph "
            f"capture and cannot be {reuse_kind}"
        )


def _workspace_buffer(
    workspace: Optional[MSASparseAttentionWorkspace],
    name: str,
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
    zero: bool = False,
) -> torch.Tensor:
    if workspace is None:
        tensor = torch.empty(shape, dtype=dtype, device=device)
    else:
        tensor = workspace._buffers.get(name)
        valid = (
            tensor is not None
            and tensor.device == device
            and tensor.dtype == dtype
            and tuple(tensor.shape) == shape
            and tensor.is_contiguous()
        )
        if not valid:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    f"MSASparseAttentionWorkspace buffer {name!r} is not warmed "
                    f"for shape {shape} and dtype {dtype}"
                )
            tensor = torch.empty(shape, dtype=dtype, device=device)
            workspace._buffers[name] = tensor
    if zero:
        tensor.zero_()
    return tensor


def _tensor_signature(tensor: torch.Tensor) -> tuple:
    return (
        tensor.data_ptr(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
    )


def _launch_signature(
    *,
    variant: str,
    target: str,
    tensors: tuple[torch.Tensor, ...],
    scalars: tuple,
    grid: tuple[int, int, int],
) -> tuple:
    return (
        variant,
        target,
        tuple(_tensor_signature(tensor) for tensor in tensors),
        scalars,
        grid,
    )


def _check_warmed_launch(
    workspace: Optional[MSASparseAttentionWorkspace],
    signature: tuple,
    *,
    capturing: bool,
) -> None:
    if capturing and (workspace is None or signature not in workspace._warmed_launches):
        raise RuntimeError(
            "MSA CUDA graph capture requires an explicit "
            "MSASparseAttentionWorkspace warmed by an eager call with the "
            "exact tensors, options, and capture stream"
        )


def _record_successful_launch(
    workspace: Optional[MSASparseAttentionWorkspace],
    signature: tuple,
    *,
    capturing: bool,
) -> None:
    if workspace is None:
        return
    if capturing:
        workspace._captured = True
    else:
        workspace._warmed_launches.add(signature)


def _require_cuda_i32(
    value,
    *,
    device: torch.device,
    name: str,
    length: Optional[int] = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"{name} must be a preallocated CUDA int32 tensor during graph capture"
            )
        value = torch.as_tensor(value, dtype=torch.int32, device=device)
    elif (
        value.device != device
        or value.dtype != torch.int32
        or not value.is_contiguous()
    ):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"{name} must already be contiguous CUDA int32 on {device} "
                "during graph capture"
            )
        value = value.to(
            device=device, dtype=torch.int32, non_blocking=True
        ).contiguous()
    if value.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if length is not None and value.numel() != length:
        raise ValueError(f"{name} must contain {length} entries")
    return value


def _explicit_q_offsets(
    q_offset,
    *,
    batch_size: int,
    device: torch.device,
    workspace: Optional[MSASparseAttentionWorkspace],
    name: str,
) -> torch.Tensor:
    if isinstance(q_offset, int):
        offsets = _workspace_buffer(
            workspace,
            name,
            (batch_size,),
            dtype=torch.int32,
            device=device,
        )
        offsets.fill_(q_offset)
        return offsets
    return _require_cuda_i32(
        q_offset,
        device=device,
        name="q_offset",
        length=batch_size,
    )


def _cumulative_kv_lengths(
    kv_lens: torch.Tensor,
    *,
    workspace: Optional[MSASparseAttentionWorkspace],
    name: str,
) -> torch.Tensor:
    cu_k = _workspace_buffer(
        workspace,
        name,
        (kv_lens.numel() + 1,),
        dtype=torch.int32,
        device=kv_lens.device,
    )
    cu_k[0].zero_()
    torch.cumsum(kv_lens, dim=0, dtype=torch.int32, out=cu_k[1:])
    return cu_k


def _validate_scale_arguments(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_scale,
    v_scale,
    k_global_scale,
    v_global_scale,
    allow_uniform_fp8: bool,
) -> tuple[float, float]:
    if k.dtype == torch.uint8:
        raise NotImplementedError(
            "NVFP4 K/V is not supported by MSA on compute capability 10.0/10.3"
        )
    if k_scale is not None or v_scale is not None:
        raise NotImplementedError(
            "tensor K/V scales are not supported by MSA on compute capability 10.0/10.3"
        )
    uniform_fp8 = q.dtype == k.dtype == v.dtype == torch.float8_e4m3fn
    if (k_global_scale is not None or v_global_scale is not None) and not (
        allow_uniform_fp8 and uniform_fp8
    ):
        raise NotImplementedError("global K/V scales require uniform FP8 Q/K/V decode")
    return (
        1.0 if k_global_scale is None else float(k_global_scale),
        1.0 if v_global_scale is None else float(v_global_scale),
    )


def _validate_attention_tensors(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_indices: torch.Tensor,
) -> tuple[int, int, int, int]:
    if not isinstance(q, torch.Tensor) or not q.is_cuda:
        raise ValueError("q must be a CUDA tensor")
    if q.dtype not in (torch.bfloat16, torch.float16, torch.float8_e4m3fn):
        raise ValueError(f"q must be bf16/fp16/fp8, got {q.dtype}")
    if q.ndim != 3 or q.shape[2] != _HEAD_DIM:
        raise ValueError(f"q must have shape (total_q, num_q_heads, {_HEAD_DIM})")
    if not q.is_contiguous():
        raise ValueError("q must be contiguous")
    total_q, num_q_heads, _ = (int(value) for value in q.shape)
    if total_q <= 0 or num_q_heads <= 0:
        raise ValueError("q must contain at least one query and one head")

    if not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor):
        raise ValueError("k and v must be CUDA tensors")
    if k.ndim not in (3, 4):
        raise ValueError("k/v must use a flat 3D or paged 4D layout")
    if k.device != q.device or v.device != q.device:
        raise ValueError("k/v must be on the same device as q")
    if k.shape != v.shape or k.dtype != v.dtype:
        raise ValueError("k/v must have the same shape and dtype")
    if not k.is_contiguous() or not v.is_contiguous():
        if k.ndim == 4:
            raise ValueError(
                "MSA on compute capability 10.0/10.3 does not directly support "
                "K/V views split from a packed paged cache; pass separate "
                "contiguous K and V tensors (implicit copies are not performed)"
            )
        raise ValueError("k/v must be contiguous")
    fp8_kv = k.dtype == torch.float8_e4m3fn
    if fp8_kv:
        if q.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
            raise NotImplementedError(
                "FP8 K/V with FP16 Q is not supported on compute capability 10.0/10.3"
            )
        if q.dtype == torch.float8_e4m3fn and v.dtype != torch.float8_e4m3fn:
            raise ValueError("FP8 Q requires uniform FP8 K/V")
    elif k.dtype != q.dtype:
        raise ValueError("dense k/v dtype must match q; FP8 K/V requires BF16 Q")
    num_kv_heads = int(k.shape[1])
    if num_kv_heads <= 0 or num_q_heads % num_kv_heads:
        raise ValueError("num_q_heads must be a multiple of num_kv_heads")
    group_size = num_q_heads // num_kv_heads
    if not 0 < group_size <= 16:
        raise ValueError("the GQA group size must be in [1, 16]")

    if (
        not isinstance(q2k_indices, torch.Tensor)
        or q2k_indices.device != q.device
        or q2k_indices.dtype != torch.int32
        or q2k_indices.ndim != 3
        or tuple(q2k_indices.shape[:2]) != (num_kv_heads, total_q)
        or not q2k_indices.is_contiguous()
    ):
        raise ValueError(
            "q2k_indices must be contiguous CUDA int32 with shape "
            "(num_kv_heads, total_q, topk)"
        )
    topk = int(q2k_indices.shape[2])
    if topk not in _SUPPORTED_ATTENTION_TOPK:
        raise ValueError(
            "Blackwell MSA sparse attention requires topk in {4, 8, 16, 32}"
        )
    return total_q, num_q_heads, num_kv_heads, group_size


def _prepare_layout(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    page_table: Optional[torch.Tensor],
    seqused_k: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    batch_size: int,
    prefill: bool,
    workspace: Optional[MSASparseAttentionWorkspace],
) -> tuple[bool, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    paged = page_table is not None
    if paged:
        if seqused_k is None:
            raise ValueError("paged K/V requires seqused_k")
        if k.ndim != 4 or k.shape[2] != _BLOCK_SIZE or k.shape[3] != _HEAD_DIM:
            raise ValueError(
                "paged k/v must have shape (num_pages, num_kv_heads, 128, 128)"
            )
        if (
            not isinstance(page_table, torch.Tensor)
            or page_table.device != q.device
            or page_table.dtype != torch.int32
            or page_table.ndim != 2
            or page_table.shape[0] != batch_size
            or not page_table.is_contiguous()
        ):
            raise ValueError(
                "page_table must be contiguous CUDA int32 with shape "
                "(batch_size, max_pages)"
            )
        kv_lens = _require_cuda_i32(
            seqused_k,
            device=q.device,
            name="seqused_k",
            length=batch_size,
        )
        if cu_seqlens_k is None:
            cu_k = (
                _cumulative_kv_lengths(
                    kv_lens,
                    workspace=workspace,
                    name="prefill_cu_seqlens_k",
                )
                if prefill
                else kv_lens
            )
        else:
            cu_k = _require_cuda_i32(
                cu_seqlens_k,
                device=q.device,
                name="cu_seqlens_k",
                length=batch_size + 1,
            )
        return True, cu_k, kv_lens, page_table, int(page_table.shape[1])

    if k.ndim != 3 or k.shape[2] != _HEAD_DIM:
        raise ValueError("flat k/v must have shape (total_k, num_kv_heads, 128)")
    if cu_seqlens_k is None:
        raise ValueError("flat K/V requires cu_seqlens_k")
    cu_k = _require_cuda_i32(
        cu_seqlens_k,
        device=q.device,
        name="cu_seqlens_k",
        length=batch_size + 1,
    )
    return False, cu_k, cu_k, q.reshape(-1).view(torch.int32), 0


def _prefill_variant(
    *,
    q_dtype: torch.dtype,
    k_dtype: torch.dtype,
    paged: bool,
    folded_gqa_group: int,
    causal: bool,
    max_pages: int,
) -> "BlackwellMSAVariant":
    layout = "paged" if paged else "flat"
    if folded_gqa_group == 8:
        return f"prefill_union_bf16_gqa8_{layout}"  # type: ignore[return-value]
    if folded_gqa_group == 16:
        if paged:
            suffix = (
                "causal_mask64"
                if causal and max_pages <= 64
                else "causal_large"
                if causal
                else "noncausal"
            )
            return f"prefill_union_bf16_gqa16_paged_{suffix}"  # type: ignore[return-value]
        return f"prefill_union_bf16_gqa16_{layout}"  # type: ignore[return-value]
    if k_dtype == torch.float8_e4m3fn:
        return f"prefill_union_bf16_query_fp8_kv_{layout}"  # type: ignore[return-value]
    dtype_name = "fp16" if q_dtype == torch.float16 else "bf16"
    return f"prefill_union_{dtype_name}_{layout}"  # type: ignore[return-value]


def _decode_variant(
    *,
    q_dtype: torch.dtype,
    k_dtype: torch.dtype,
    paged: bool,
) -> "BlackwellMSAVariant":
    layout = "paged" if paged else "flat"
    if k_dtype == torch.float8_e4m3fn:
        return f"decode_m16_bf16_query_fp8_kv_{layout}"  # type: ignore[return-value]
    dtype_name = "fp16" if q_dtype == torch.float16 else "bf16"
    return f"decode_m16_{dtype_name}_{layout}"  # type: ignore[return-value]


def _exact_non16_decode_variant(
    *,
    requested_schedule: str,
    capturing: bool,
    paged: bool,
    force_fused: Optional[bool],
    causal: bool,
    q_offset_is_none: bool,
    q_dtype: torch.dtype,
    k_dtype: torch.dtype,
    batch_size: int,
    total_q: int,
    seqlen_q: int,
    num_q_heads: int,
    num_kv_heads: int,
    topk: int,
    k_outer_dim: int,
    max_pages: int,
) -> Optional["BlackwellMSAVariant"]:
    """Select one of the two exact eager non-TopK16 decode routes."""

    common = (
        requested_schedule == ""
        and not capturing
        and paged
        and force_fused is True
        and causal
        and q_offset_is_none
        and q_dtype == torch.bfloat16
        and k_dtype == torch.bfloat16
    )
    if not common:
        return None
    if (
        topk == 32
        and batch_size == 64
        and total_q == 512
        and seqlen_q == 8
        and num_q_heads == 64
        and num_kv_heads == 4
        and k_outer_dim == 32768
        and max_pages == 512
    ):
        return "decode_m16_bf16_paged_topk32"
    if (
        topk == 4
        and batch_size == 2
        and total_q == 2
        and seqlen_q == 1
        and num_q_heads == 8
        and num_kv_heads == 1
        and k_outer_dim == 6
        and max_pages == 3
    ):
        return "decode_m16_bf16_paged_topk4_exact512"
    return None


def _is_exact_fp8_topk8_qagg_prefill(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_indices: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    paged: bool,
    batch_size: int,
    causal: bool,
    q_offset_is_none: bool,
    softmax_scale: Optional[float],
    return_temperature_lse: bool,
    lse_temperature_scale: float,
    requested_schedule: str,
    capturing: bool,
) -> bool:
    return bool(
        not capturing
        and requested_schedule == ""
        and not paged
        and batch_size == 3
        and q.dtype == torch.bfloat16
        and tuple(q.shape) == (3072, 32, _HEAD_DIM)
        and k.dtype == v.dtype == torch.float8_e4m3fn
        and tuple(k.shape) == tuple(v.shape) == (24576, 2, _HEAD_DIM)
        and tuple(q2k_indices.shape) == (2, 3072, 8)
        and tuple(cu_q.shape) == tuple(cu_k.shape) == (4,)
        and causal
        and q_offset_is_none
        and softmax_scale is None
        and return_temperature_lse
        and lse_temperature_scale == 1.0
    )


def _is_exact_bf16_topk4_qload4_prefill(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_indices: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    page_table: torch.Tensor,
    kv_lens: torch.Tensor,
    paged: bool,
    batch_size: int,
    causal: bool,
    q_offset_is_none: bool,
    softmax_scale: Optional[float],
    return_temperature_lse: bool,
    lse_temperature_scale: float,
    requested_schedule: str,
) -> bool:
    return bool(
        requested_schedule == ""
        and paged
        and batch_size == 3
        and q.dtype == k.dtype == v.dtype == torch.bfloat16
        and tuple(q.shape) == (12288, 8, _HEAD_DIM)
        and tuple(k.shape) == tuple(v.shape) == (192, 2, _BLOCK_SIZE, _HEAD_DIM)
        and tuple(q2k_indices.shape) == (2, 12288, 4)
        and tuple(cu_q.shape) == tuple(cu_k.shape) == (4,)
        and tuple(page_table.shape) == (3, 64)
        and tuple(kv_lens.shape) == (3,)
        and causal
        and q_offset_is_none
        and softmax_scale is None
        and (not return_temperature_lse or lse_temperature_scale == 1.0)
    )


def _resolve_fp8_q1_schedule(
    *,
    requested: str,
    capturing: bool,
    paged: bool,
    force_fused: Optional[bool],
    causal: bool,
    q_offset_is_none: bool,
    q_dtype: torch.dtype,
    k_dtype: torch.dtype,
    batch_size: int,
    total_q: int,
    seqlen_q: int,
    num_q_heads: int,
    num_kv_heads: int,
    topk: int,
    k_outer_dim: int,
    max_pages: int,
) -> str:
    """Select only the two default FP8-KV Q1 serving specializations."""

    if requested:
        return requested
    common = (
        not capturing
        and force_fused is True
        and causal
        and q_offset_is_none
        and q_dtype == torch.bfloat16
        and k_dtype == torch.float8_e4m3fn
        and batch_size == total_q
        and seqlen_q == 1
        and num_q_heads == 64
        and num_kv_heads == 4
        and topk == _ATTENTION_TOPK
    )
    if not common:
        return ""
    if paged and batch_size == 128 and k_outer_dim == 4096 and max_pages == 32:
        return "q1_paged_xform2"
    if not paged and batch_size == 32 and k_outer_dim == 262144:
        return "q1_flat_xform2"
    return ""


def _uniform_fp8_decode_grid(
    *, total_work_items: int, num_sms: int, seqlen_q: int
) -> int:
    physical_grid = min(total_work_items, num_sms)
    default_waves = (total_work_items + physical_grid - 1) // physical_grid
    even_waves = (
        total_work_items + _UNIFORM_FP8_EVEN_WAVE_GRID - 1
    ) // _UNIFORM_FP8_EVEN_WAVE_GRID
    if (
        seqlen_q >= 4
        and num_sms >= _UNIFORM_FP8_EVEN_WAVE_GRID
        and total_work_items % _UNIFORM_FP8_EVEN_WAVE_GRID == 0
        and even_waves == default_waves
    ):
        return _UNIFORM_FP8_EVEN_WAVE_GRID
    return physical_grid


def _should_use_long_prefill(
    *,
    requested_schedule: str,
    batch_size: int,
    total_q: int,
    paged: bool,
    group_size: int,
    max_pages: int,
    k_outer_dim: int,
    q_dtype: torch.dtype,
    k_dtype: torch.dtype,
    v_dtype: torch.dtype,
    causal: bool,
    q_offset_is_none: bool,
    return_temperature_lse: bool,
    lse_temperature_scale: float,
) -> bool:
    return bool(
        requested_schedule != "m64"
        and batch_size == 1
        and total_q >= 8192
        and (
            (
                paged
                and (
                    (group_size == 8 and max_pages >= 64)
                    or (group_size == 16 and max_pages > 64)
                )
            )
            or (not paged and group_size == 16 and k_outer_dim >= 8192)
        )
        and q_dtype == k_dtype == v_dtype == torch.bfloat16
        and causal
        and q_offset_is_none
        and (not return_temperature_lse or lse_temperature_scale == 1.0)
    )


def _long_plan_signature(
    q2k_indices: torch.Tensor,
    *,
    total_k: int,
    num_sms: int,
    group_size: int,
    paged: bool,
) -> tuple:
    return (
        q2k_indices.device.type,
        q2k_indices.device.index,
        int(q2k_indices.data_ptr()),
        int(q2k_indices._version),
        tuple(q2k_indices.shape),
        total_k,
        num_sms,
        group_size,
        paged,
    )


def _long_state_tensor(
    state: dict,
    name: str,
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    tensor = state.get(name)
    valid = (
        isinstance(tensor, torch.Tensor)
        and tuple(tensor.shape) == shape
        and tensor.dtype == dtype
        and tensor.device == device
        and tensor.is_contiguous()
    )
    if not valid:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"long-prefill workspace entry {name!r} must be warmed before capture"
            )
        tensor = torch.empty(shape, dtype=dtype, device=device)
        state[name] = tensor
    return tensor


def _build_long_prefill_plan(
    q2k_indices: torch.Tensor,
    *,
    total_k: int,
    num_sms: int,
    group_size: int,
    paged: bool,
) -> dict:
    num_kv_heads, total_q, topk = (int(value) for value in q2k_indices.shape)
    if topk != _ATTENTION_TOPK:
        raise ValueError("long prefill requires topk=16")
    if group_size not in {8, 16} or (not paged and group_size != 16):
        raise ValueError("long prefill requires paged GQA8/GQA16 or flat GQA16")
    if total_q > _LONG_QSPLIT_Q_MASK:
        raise ValueError("long prefill exceeds the packed query-index range")
    total_rows = (total_k + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    nnz_per_head = total_q * topk
    q_tokens_per_group = 128 // group_size
    work_group_cap = _LONG_WORK_GROUP_BUCKETS[0]
    if paged:
        total_groups_upper = (
            num_kv_heads * total_q * topk + q_tokens_per_group - 1
        ) // q_tokens_per_group
        target = max(1, (total_groups_upper + 2 * num_sms - 1) // (2 * num_sms))
        work_group_cap = min(128, 1 << (target - 1).bit_length())
        if group_size == 8 and total_rows == 64:
            work_group_cap = max(work_group_cap, 64)
    buckets = tuple(
        value for value in _LONG_WORK_GROUP_BUCKETS if value <= work_group_cap
    )

    row_ptr = torch.zeros(
        (num_kv_heads, total_rows + 1), dtype=torch.int32, device=q2k_indices.device
    )
    qsplit = torch.full(
        (num_kv_heads, nnz_per_head), -1, dtype=torch.int32, device=q2k_indices.device
    )
    split_counts_by_head = ((q2k_indices >= 0) & (q2k_indices < total_rows)).sum(
        dim=2, dtype=torch.int32
    )
    counts_host: list[list[int]] = []
    for head in range(num_kv_heads):
        flat = q2k_indices[head].reshape(-1)
        valid = (flat >= 0) & (flat < total_rows)
        positions = torch.nonzero(valid, as_tuple=False).flatten()
        blocks = flat.index_select(0, positions)
        counts = torch.bincount(blocks.to(torch.int64), minlength=total_rows).to(
            torch.int32
        )
        row_ptr[head, 1:] = torch.cumsum(counts, dim=0, dtype=torch.int32)
        counts_host.append([int(value) for value in counts.cpu().tolist()])
        order = torch.argsort(blocks, stable=True)
        sorted_positions = positions.index_select(0, order)
        q_indices = torch.div(sorted_positions, topk, rounding_mode="floor")
        slots = sorted_positions - q_indices * topk
        packed = q_indices.to(torch.int32) | (
            slots.to(torch.int32) << _LONG_QSPLIT_SLOT_SHIFT
        )
        packed |= (split_counts_by_head[head].index_select(0, q_indices) == 1).to(
            torch.int32
        ) << _LONG_QSPLIT_SINGLE_SHIFT
        qsplit[head, : packed.numel()] = packed

    work: list[tuple[int, tuple[int, int, int, int, int, int]]] = []
    for head, counts in enumerate(counts_host):
        for kv_block, row_count in enumerate(counts):
            q_begin = 0
            remaining = row_count
            for group_count in buckets:
                capacity = group_count * q_tokens_per_group
                while (
                    remaining + q_tokens_per_group - 1
                ) // q_tokens_per_group >= group_count:
                    q_count = min(capacity, remaining)
                    work.append(
                        (group_count, (head, kv_block, q_begin, q_count, 0, kv_block))
                    )
                    q_begin += q_count
                    remaining -= q_count
            if remaining:
                raise AssertionError("long-prefill work decomposition failed")
    if not work:
        raise ValueError("long prefill requires at least one selected edge")
    work.sort(key=lambda item: item[0], reverse=True)
    metadata = torch.tensor(
        [entry for _group, entry in work], dtype=torch.int32, device=q2k_indices.device
    ).contiguous()
    counts_by_group = [0] * 129
    for group_count, _entry in work:
        counts_by_group[group_count] += 1
    running = 0
    end_by_group: dict[int, int] = {}
    for group_count in range(128, 1, -1):
        running += counts_by_group[group_count]
        end_by_group[group_count] = running
    return {
        "scheduler_metadata": metadata,
        "row_ptr": row_ptr.contiguous(),
        "qsplit": qsplit.contiguous(),
        "split_counts": split_counts_by_head.transpose(0, 1).contiguous(),
        "group_segment_ends": tuple(
            end_by_group[value] for value in _LONG_GROUP_BOUNDARIES
        ),
        "work_count": len(work),
        "total_rows": total_rows,
        "nnz_per_head": nnz_per_head,
    }


def _get_long_prefill_state(
    *,
    workspace: Optional[MSASparseAttentionWorkspace],
    q2k_indices: torch.Tensor,
    total_k: int,
    num_sms: int,
    group_size: int,
    paged: bool,
) -> dict:
    signature = _long_plan_signature(
        q2k_indices,
        total_k=total_k,
        num_sms=num_sms,
        group_size=group_size,
        paged=paged,
    )
    if workspace is not None:
        state = workspace._long_prefill_state
    else:
        with _implicit_long_prefill_states_lock:
            state = _implicit_long_prefill_states.pop(signature, {})
            _implicit_long_prefill_states.clear()
            _implicit_long_prefill_states[signature] = state
    if state.get("signature") != signature:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "long-prefill plan must be warmed before CUDA graph capture"
            )
        state.clear()
        state["signature"] = signature
        state["plan"] = _build_long_prefill_plan(
            q2k_indices,
            total_k=total_k,
            num_sms=num_sms,
            group_size=group_size,
            paged=paged,
        )
        state["q2k_owner"] = q2k_indices
    return state


def _run_long_prefill_modules(
    *,
    target: "BlackwellMSATarget",
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    temperature_lse: torch.Tensor,
    q2k_indices: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    q_offsets: torch.Tensor,
    kv_lens: torch.Tensor,
    page_table: torch.Tensor,
    paged: bool,
    group_size: int,
    max_pages: int,
    softmax_scale_log2: float,
    lse_temperature_scale: float,
    return_softmax_lse: bool,
    return_temperature_lse: bool,
    stream_ptr: int,
    workspace: Optional[MSASparseAttentionWorkspace],
    capturing: bool,
) -> None:
    total_q, num_q_heads, _ = (int(value) for value in q.shape)
    num_kv_heads = int(k.shape[1])
    total_k = max_pages * _BLOCK_SIZE if paged else int(k.shape[0])
    num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    state = _get_long_prefill_state(
        workspace=workspace,
        q2k_indices=q2k_indices,
        total_k=total_k,
        num_sms=num_sms,
        group_size=group_size,
        paged=paged,
    )
    plan = state["plan"]
    partial_o = _long_state_tensor(
        state,
        "partial_o",
        (_ATTENTION_TOPK, total_q, num_q_heads, _HEAD_DIM),
        dtype=torch.uint8,
        device=q.device,
    )
    scale_shape: tuple[int, ...]
    if paged:
        scale_shape = (
            _ATTENTION_TOPK,
            total_q,
            num_q_heads,
            _LONG_PARTIAL_SEGMENT_COUNT,
        )
        scale_dtype = torch.bfloat16
    else:
        scale_shape = (2, _ATTENTION_TOPK, total_q, num_q_heads)
        scale_dtype = torch.float32
    partial_scale = _long_state_tensor(
        state, "partial_scale", scale_shape, dtype=scale_dtype, device=q.device
    )
    partial_lse = _long_state_tensor(
        state,
        "partial_lse",
        (_ATTENTION_TOPK, total_q, num_q_heads),
        dtype=torch.float32,
        device=q.device,
    )
    partial_temperature_lse = partial_lse
    if return_temperature_lse:
        partial_temperature_lse = _long_state_tensor(
            state,
            "partial_temperature_lse",
            (_ATTENTION_TOPK, total_q, num_q_heads),
            dtype=torch.float32,
            device=q.device,
        )
    arch_suffix = "sm103" if target == "sm103a" else "sm100"
    if paged:
        direct = target == "sm100a" and group_size == 16 and max_pages == 8192
        forward_variant = cast(
            "BlackwellMSAVariant",
            (
                "long_prefill_paged_bf16_gqa16_direct_group_sm100"
                if direct
                else f"long_prefill_paged_bf16_gqa{group_size}_{arch_suffix}"
            ),
        )
        reduce_variant = cast(
            "BlackwellMSAVariant", f"long_prefill_reduce_paged_bf16_gqa{group_size}"
        )
    else:
        forward_variant = cast(
            "BlackwellMSAVariant", f"long_prefill_flat_bf16_gqa16_{arch_suffix}"
        )
        reduce_variant = "long_prefill_reduce_flat_bf16_gqa16"
    forward_grid = (int(plan["work_count"]), 1, 1)
    reduce_grid = ((total_q * num_q_heads + 31) // 32, 1, 1)
    page_table_arg = page_table if paged else q2k_indices.reshape(-1)
    forward_tensors = (
        q,
        k,
        v,
        plan["scheduler_metadata"],
        plan["row_ptr"],
        plan["qsplit"],
        partial_o,
        partial_scale,
        partial_lse,
        partial_temperature_lse,
        out,
        cu_q,
        cu_k,
        q_offsets,
        kv_lens,
        page_table_arg,
    )
    forward_scalars = (
        *plan["group_segment_ends"],
        total_q,
        num_q_heads,
        num_kv_heads,
        int(plan["total_rows"]),
        int(plan["nnz_per_head"]),
        int(plan["work_count"]),
        int(plan["work_count"]),
        _ATTENTION_TOPK,
        max_pages if paged else 0,
        1,
        1,
        softmax_scale_log2,
        lse_temperature_scale,
        int(return_temperature_lse),
    )
    reduce_tensors = (
        partial_o,
        partial_scale,
        partial_lse,
        partial_temperature_lse,
        plan["split_counts"],
        out,
        lse,
        temperature_lse,
    )
    reduce_scalars = (
        total_q,
        num_q_heads,
        num_kv_heads,
        group_size,
        _ATTENTION_TOPK,
        int(return_softmax_lse or return_temperature_lse),
        int(return_temperature_lse),
    )
    signature = _launch_signature(
        variant=f"{forward_variant}+{reduce_variant}",
        target=target,
        tensors=(*forward_tensors, *reduce_tensors),
        scalars=(*forward_scalars, *reduce_scalars),
        grid=forward_grid,
    )
    _check_warmed_launch(workspace, signature, capturing=capturing)
    _get_module(forward_variant, target).run(
        *forward_tensors, *forward_scalars, *forward_grid, stream_ptr
    )
    _get_module(reduce_variant, target).run(
        *reduce_tensors, *reduce_scalars, *reduce_grid, stream_ptr
    )
    _record_successful_launch(workspace, signature, capturing=capturing)


def _reverse_prefill_state(
    workspace: Optional[MSASparseAttentionWorkspace], route: str
) -> dict:
    states = (
        workspace._reverse_prefill_states
        if workspace is not None
        else _implicit_reverse_prefill_states
    )
    return states.setdefault(route, {})


def _run_exact_fp8_topk8_qagg_prefill(
    *,
    target: "BlackwellMSATarget",
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_indices: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    stream_ptr: int,
    workspace: Optional[MSASparseAttentionWorkspace],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch the exact eager TopK8 producer/reducer pair."""

    from ._blackwell_sm100_reverse_plan import prepare_fp8_topk8_qagg_plan

    route = "fp8_topk8_qagg_pdl"
    context = (
        nullcontext()
        if workspace is not None
        else _implicit_reverse_prefill_states_lock
    )
    with context:
        state = _reverse_prefill_state(workspace, route)
        try:
            plan = prepare_fp8_topk8_qagg_plan(
                q2k_indices,
                cu_q,
                cu_k,
                sm_count=torch.cuda.get_device_properties(
                    q.device
                ).multi_processor_count,
                stream_id=stream_ptr,
                state=state,
            )
            geometry = plan["geometry"]
            completion_counts = _long_state_tensor(
                state,
                "completion_counts",
                (384,),
                dtype=torch.uint32,
                device=q.device,
            )
            if "launches_completed" not in state:
                completion_counts.zero_()
                state["launches_completed"] = 0
            launches_completed = int(state["launches_completed"])
            if launches_completed >= (1 << 32) - 1:
                state.clear()
                raise OverflowError(
                    "TopK8 qagg generation exhausted; the plan was invalidated"
                )
            generation = launches_completed + 1
            out = _long_state_tensor(
                state,
                "out",
                (3072, 32, _HEAD_DIM),
                dtype=torch.bfloat16,
                device=q.device,
            )
            lse = _long_state_tensor(
                state,
                "lse",
                (3072, 32),
                dtype=torch.float32,
                device=q.device,
            )
            temperature_lse = _long_state_tensor(
                state,
                "temperature_lse",
                (3072, 32),
                dtype=torch.float32,
                device=q.device,
            )
            partial_o = _long_state_tensor(
                state,
                "partial_o",
                (8, 3072, 32, _HEAD_DIM),
                dtype=torch.float8_e4m3fn,
                device=q.device,
            )
            partial_lse = _long_state_tensor(
                state,
                "partial_lse",
                (8, 3072, 32),
                dtype=torch.float32,
                device=q.device,
            )
            partial_temperature_lse = _long_state_tensor(
                state,
                "partial_temperature_lse",
                (8, 3072, 32),
                dtype=torch.float32,
                device=q.device,
            )
            i32_dummy = _long_state_tensor(
                state,
                "i32_dummy",
                (1,),
                dtype=torch.int32,
                device=q.device,
            )
            if "i32_dummy_initialized" not in state:
                i32_dummy.zero_()
                state["i32_dummy_initialized"] = True
            producer_variant: BlackwellMSAVariant = (
                "reverse_prefill_bf16_query_fp8_kv_flat_topk8_qagg_pdl"
            )
            reducer_variant: BlackwellMSAVariant = (
                "reverse_prefill_bf16_query_fp8_kv_flat_topk8_qagg_pdl_reduce"
            )
            producer_tensors = (
                q,
                k.view(torch.uint8),
                v.view(torch.uint8),
                plan["scheduler_metadata"],
                plan["k2q_row_ptr"],
                plan["k2q_qsplit_indices"],
                partial_o,
                partial_lse,
                partial_temperature_lse,
                completion_counts,
                cu_q,
                cu_k,
                i32_dummy,
                i32_dummy,
                i32_dummy,
            )
            producer_scalars = (
                3072,
                32,
                2,
                int(geometry.total_rows),
                3072 * 8,
                int(geometry.schedule_capacity),
                int(geometry.work_count),
                8,
                0,
                1,
                1,
                (_HEAD_DIM**-0.5) / math.log(2.0),
                1.0,
                0,
            )
            reducer_tensors = (
                partial_o,
                partial_lse,
                partial_temperature_lse,
                plan["split_counts"],
                plan["q_order"],
                plan["contributor_work_ids"],
                completion_counts,
                out,
                lse,
                temperature_lse,
            )
            reducer_scalars = (
                3072,
                32,
                2,
                16,
                8,
                generation,
                1,
                1,
            )
            _get_module(producer_variant, target).run(
                *producer_tensors,
                *producer_scalars,
                384,
                1,
                1,
                stream_ptr,
            )
            _get_module(reducer_variant, target).run(
                *reducer_tensors,
                *reducer_scalars,
                3072,
                1,
                1,
                stream_ptr,
            )
            state["launches_completed"] = generation
            return out, lse, temperature_lse
        except BaseException:
            state.clear()
            raise


def _exact_topk4_launch_parts(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    kv_lens: torch.Tensor,
    page_table: torch.Tensor,
    return_softmax_lse: bool,
    return_temperature_lse: bool,
    state: dict,
) -> tuple[
    tuple,
    tuple,
    tuple[int, int, int],
    tuple,
    tuple,
    tuple[int, int, int],
    tuple,
]:
    plan = state
    geometry = plan["geometry"]
    out = _long_state_tensor(
        state,
        "out",
        (12288, 8, _HEAD_DIM),
        dtype=torch.bfloat16,
        device=q.device,
    )
    lse = _long_state_tensor(
        state,
        "lse",
        (12288, 8),
        dtype=torch.float32,
        device=q.device,
    )
    temperature_lse = _long_state_tensor(
        state,
        "temperature_lse",
        (12288, 8),
        dtype=torch.float32,
        device=q.device,
    )
    partial_o = _long_state_tensor(
        state,
        "partial_o",
        (4, 12288, 8, _HEAD_DIM),
        dtype=torch.uint8,
        device=q.device,
    )
    partial_scale = _long_state_tensor(
        state,
        "partial_scale",
        (4, 12288, 8, 4),
        dtype=torch.float32,
        device=q.device,
    )
    partial_lse = _long_state_tensor(
        state,
        "partial_lse",
        (4, 12288, 8),
        dtype=torch.float32,
        device=q.device,
    )
    partial_temperature_lse = _long_state_tensor(
        state,
        "partial_temperature_lse",
        (4, 12288, 8),
        dtype=torch.float32,
        device=q.device,
    )
    producer_tensors = (
        q,
        k,
        v,
        plan["scheduler_metadata"],
        plan["k2q_row_ptr"],
        plan["k2q_qsplit_indices"],
        partial_o,
        partial_scale,
        partial_lse,
        partial_temperature_lse,
        cu_q,
        cu_k,
        cu_k,
        kv_lens,
        page_table,
    )
    producer_scalars = (
        *plan["group_segment_ends"],
        12288,
        8,
        2,
        int(geometry.total_rows),
        12288 * 4,
        int(geometry.schedule_capacity),
        int(geometry.work_count),
        4,
        64,
        1,
        1,
        (_HEAD_DIM**-0.5) / math.log(2.0),
        1.0,
        int(return_temperature_lse),
    )
    reducer_tensors = (
        partial_o,
        partial_scale,
        partial_lse,
        partial_temperature_lse,
        plan["split_counts"],
        out,
        lse,
        temperature_lse,
    )
    reducer_scalars = (
        12288,
        8,
        2,
        4,
        4,
        int(return_softmax_lse or return_temperature_lse),
        int(return_temperature_lse),
    )
    return (
        producer_tensors,
        producer_scalars,
        (int(geometry.work_count), 1, 1),
        reducer_tensors,
        reducer_scalars,
        (3072, 1, 1),
        (out, lse, temperature_lse),
    )


def _enqueue_exact_topk4_pair(
    *,
    target: "BlackwellMSATarget",
    parts: tuple,
    stream_ptr: int,
) -> None:
    (
        producer_tensors,
        producer_scalars,
        producer_grid,
        reducer_tensors,
        reducer_scalars,
        reducer_grid,
        _outputs,
    ) = parts
    producer_variant: BlackwellMSAVariant = "reverse_prefill_bf16_paged_topk4_qload4"
    reducer_variant: BlackwellMSAVariant = (
        "reverse_prefill_bf16_paged_topk4_qload4_const4_reduce"
    )
    _get_module(producer_variant, target).run(
        *producer_tensors,
        *producer_scalars,
        *producer_grid,
        stream_ptr,
    )
    _get_module(reducer_variant, target).run(
        *reducer_tensors,
        *reducer_scalars,
        *reducer_grid,
        stream_ptr,
    )


def _run_exact_bf16_topk4_qload4_prefill(
    *,
    target: "BlackwellMSATarget",
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_indices: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    kv_lens: torch.Tensor,
    page_table: torch.Tensor,
    return_softmax_lse: bool,
    return_temperature_lse: bool,
    stream_ptr: int,
    workspace: Optional[MSASparseAttentionWorkspace],
    capturing: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch or replay the exact paged TopK4 producer/reducer pair."""

    from ._blackwell_sm100_reverse_plan import prepare_bf16_paged_topk4_plan

    route = "bf16_paged_topk4_qload4"
    context = (
        nullcontext()
        if workspace is not None
        else _implicit_reverse_prefill_states_lock
    )
    with context:
        state = _reverse_prefill_state(workspace, route)
        try:
            prepare_bf16_paged_topk4_plan(
                q2k_indices,
                cu_q,
                cu_k,
                page_table,
                kv_lens,
                sm_count=torch.cuda.get_device_properties(
                    q.device
                ).multi_processor_count,
                stream_id=stream_ptr,
                state=state,
            )
            parts = _exact_topk4_launch_parts(
                q=q,
                k=k,
                v=v,
                cu_q=cu_q,
                cu_k=cu_k,
                kv_lens=kv_lens,
                page_table=page_table,
                return_softmax_lse=return_softmax_lse,
                return_temperature_lse=return_temperature_lse,
                state=state,
            )
            outputs = parts[-1]
            producer_tensors, producer_scalars, producer_grid = parts[:3]
            reducer_tensors, reducer_scalars, reducer_grid = parts[3:6]
            signature = _launch_signature(
                variant="reverse_prefill_bf16_paged_topk4_qload4_graph",
                target=target,
                tensors=(*producer_tensors, *reducer_tensors),
                scalars=(*producer_scalars, *reducer_scalars),
                grid=producer_grid,
            )
            _check_warmed_launch(workspace, signature, capturing=capturing)
            if capturing:
                _enqueue_exact_topk4_pair(
                    target=target, parts=parts, stream_ptr=stream_ptr
                )
            else:
                graph_state = state.get("graph_state")
                graph_signature = (
                    signature,
                    reducer_grid,
                    tuple(_tensor_signature(tensor) for tensor in outputs),
                )
                if (
                    not isinstance(graph_state, dict)
                    or graph_state.get("signature") != graph_signature
                ):
                    _enqueue_exact_topk4_pair(
                        target=target, parts=parts, stream_ptr=stream_ptr
                    )
                    current_stream = torch.cuda.current_stream(q.device)
                    capture_stream = torch.cuda.Stream(device=q.device)
                    capture_stream.wait_stream(current_stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=capture_stream):
                        _enqueue_exact_topk4_pair(
                            target=target,
                            parts=parts,
                            stream_ptr=int(capture_stream.cuda_stream),
                        )
                    graph_state = {
                        "signature": graph_signature,
                        "graph": graph,
                        "capture_stream": capture_stream,
                        "keepalive": parts,
                    }
                    state["graph_state"] = graph_state
                graph_state["graph"].replay()
            _record_successful_launch(workspace, signature, capturing=capturing)
            return outputs
        except BaseException:
            state.clear()
            raise


def _run_fp8_direct_module(
    *,
    schedule: str,
    target: "BlackwellMSATarget",
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    q2k_indices: torch.Tensor,
    cu_k: torch.Tensor,
    q_offsets: torch.Tensor,
    kv_lens: torch.Tensor,
    page_table: torch.Tensor,
    paged: bool,
    max_pages: int,
    seqlen_q: int,
    softmax_scale_log2: float,
    output_scale: float,
    stream_ptr: int,
    workspace: Optional[MSASparseAttentionWorkspace],
    capturing: bool,
) -> None:
    variants = {
        "q1_exact": f"decode_q1_bf16_query_fp8_kv_exact_{'paged' if paged else 'flat'}",
        "q1_flat_xform2": "decode_q1_bf16_query_fp8_kv_xform2_flat",
        "q1_paged_xform2": "decode_q1_bf16_query_fp8_kv_xform2_paged",
        "paged_uniform_fp8": "decode_uniform_fp8_qkv_paged",
    }
    variant = cast("BlackwellMSAVariant", variants[schedule])
    q_launch = q.view(torch.uint8) if schedule == "paged_uniform_fp8" else q
    page_table_arg = page_table if paged else q2k_indices.reshape(-1)
    tensors = (
        q_launch,
        k.view(torch.uint8),
        v.view(torch.uint8),
        out,
        lse,
        page_table_arg,
        cu_k,
        q2k_indices,
        q_offsets,
        kv_lens,
    )
    scalars: tuple[int | float, ...]
    if schedule == "paged_uniform_fp8":
        scalars = (
            int(q.shape[0]),
            seqlen_q,
            int(q.shape[1]),
            int(k.shape[1]),
            softmax_scale_log2,
            output_scale,
            max_pages,
        )
        total_work_items = int(q.shape[0]) * int(k.shape[1])
        grid = (
            _uniform_fp8_decode_grid(
                total_work_items=total_work_items,
                num_sms=torch.cuda.get_device_properties(
                    q.device
                ).multi_processor_count,
                seqlen_q=seqlen_q,
            ),
            1,
            1,
        )
    else:
        scalars = (int(q.shape[0]),)
        if schedule == "q1_paged_xform2":
            scalars += (int(q.shape[1]), int(k.shape[1]))
        scalars += (softmax_scale_log2, max_pages)
        grid = (int(q.shape[0]), int(k.shape[1]), 1)
    signature = _launch_signature(
        variant=variant, target=target, tensors=tensors, scalars=scalars, grid=grid
    )
    _check_warmed_launch(workspace, signature, capturing=capturing)
    _get_module(variant, target).run(*tensors, *scalars, *grid, stream_ptr)
    _record_successful_launch(workspace, signature, capturing=capturing)


def _run_prefill_module(
    *,
    variant: "BlackwellMSAVariant",
    target: "BlackwellMSATarget",
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    temperature_lse: torch.Tensor,
    q2k_indices: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    q_offsets: torch.Tensor,
    kv_lens: torch.Tensor,
    page_table: torch.Tensor,
    total_q: int,
    num_q_heads: int,
    num_kv_heads: int,
    topk: int,
    batch_size: int,
    uniform_q_len: int,
    max_pages: int,
    causal: bool,
    derive_q_offset: bool,
    softmax_scale_log2: float,
    lse_temperature_scale: float,
    return_softmax_lse: bool,
    return_temperature_lse: bool,
    grid: tuple[int, int, int],
    stream_ptr: int,
    workspace: Optional[MSASparseAttentionWorkspace],
    capturing: bool,
) -> None:
    k_launch = k.view(torch.uint8) if k.dtype == torch.float8_e4m3fn else k
    v_launch = v.view(torch.uint8) if v.dtype == torch.float8_e4m3fn else v
    tensors = (
        q,
        k_launch,
        v_launch,
        out,
        lse,
        temperature_lse,
        q2k_indices,
        cu_q,
        cu_k,
        q_offsets,
        kv_lens,
    )
    common_scalars = (
        total_q,
        num_q_heads,
        num_kv_heads,
        topk,
        batch_size,
        uniform_q_len,
        int(causal),
        int(derive_q_offset),
        softmax_scale_log2,
        lse_temperature_scale,
        int(return_softmax_lse or return_temperature_lse),
        int(return_temperature_lse),
    )
    if variant == "prefill_m64_bf16_gqa16_flat":
        signature = _launch_signature(
            variant=variant,
            target=target,
            tensors=tensors,
            scalars=common_scalars,
            grid=grid,
        )
        _check_warmed_launch(workspace, signature, capturing=capturing)
        _get_module(variant, target).run(
            *tensors,
            *common_scalars,
            *grid,
            stream_ptr,
        )
    else:
        paged_tensors = (*tensors, page_table)
        scalars = (
            total_q,
            num_q_heads,
            num_kv_heads,
            topk,
            batch_size,
            uniform_q_len,
            max_pages,
            int(causal),
            int(derive_q_offset),
            softmax_scale_log2,
            lse_temperature_scale,
            int(return_softmax_lse or return_temperature_lse),
            int(return_temperature_lse),
        )
        signature = _launch_signature(
            variant=variant,
            target=target,
            tensors=paged_tensors,
            scalars=scalars,
            grid=grid,
        )
        _check_warmed_launch(workspace, signature, capturing=capturing)
        _get_module(variant, target).run(
            *paged_tensors,
            *scalars,
            *grid,
            stream_ptr,
        )
    _record_successful_launch(workspace, signature, capturing=capturing)


def _run_decode_module(
    *,
    variant: "BlackwellMSAVariant",
    target: "BlackwellMSATarget",
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    q2k_indices: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    q_offsets: torch.Tensor,
    kv_lens: torch.Tensor,
    page_table: torch.Tensor,
    topk: int,
    max_pages: int,
    seqlen_q: int,
    softmax_scale_log2: float,
    causal: bool,
    paged: bool,
    derive_q_offset: bool,
    workspace: Optional[MSASparseAttentionWorkspace],
    capturing: bool,
    stream_ptr: int,
) -> None:
    fp8 = k.dtype == torch.float8_e4m3fn
    i32_dummy = q2k_indices.reshape(-1)
    f32_dummy = lse.reshape(-1)
    if fp8:
        q_prefill_dummy = _decode_tma_dummy(
            device=q.device,
            stream_ptr=stream_ptr,
            dtype=torch.bfloat16,
            workspace=workspace,
        )
        k_pair_dummy = q_prefill_dummy.reshape(2, 1, 128, _HEAD_DIM)
        v_pair_dummy = k_pair_dummy
        k_launch = k.view(torch.uint8)
        v_launch = v.view(torch.uint8)
    else:
        q_prefill_dummy = k.reshape(-1, 1, _HEAD_DIM)
        pair_tokens = int(q_prefill_dummy.shape[0]) // 64 * 64
        k_pair_dummy = q_prefill_dummy[:pair_tokens].reshape(-1, 1, 64, _HEAD_DIM)
        v_pair_dummy = v.reshape(-1, 1, _HEAD_DIM)[:pair_tokens].reshape(
            -1, 1, 64, _HEAD_DIM
        )
        k_launch = k
        v_launch = v
    page_table_arg = page_table if paged else i32_dummy
    total_q, num_q_heads, _ = q.shape
    num_kv_heads = k.shape[1]

    total_tasks = int(total_q) * int(num_kv_heads)
    num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    physical_ctas = min(total_tasks, num_sms)
    grid = (physical_ctas, 1, 1)
    qo_indptr = cu_q
    partial_o = f32_dummy
    partial_m = f32_dummy
    partial_d = f32_dummy
    split_completion = i32_dummy
    status = _workspace_buffer(
        workspace,
        "decode_status",
        (2,),
        dtype=torch.int32,
        device=q.device,
    )
    max_splits = 1
    max_task_claims = (total_tasks + physical_ctas - 1) // physical_ctas - 1
    attention_mode = _MODE_DECODE_ONLY
    split_policy = _SPLIT_ADAPTIVE

    tensors = (
        q,
        q_prefill_dummy,
        q_prefill_dummy,
        k_launch,
        k_pair_dummy,
        v_launch,
        v_pair_dummy,
        q.reshape(-1, _HEAD_DIM),
        out,
        partial_o,
        partial_m,
        partial_d,
        split_completion,
        lse,
        page_table_arg,
        qo_indptr,
        cu_k,
        kv_lens,
        q2k_indices,
        q_offsets,
        kv_lens,
        i32_dummy,
        i32_dummy,
        i32_dummy,
        i32_dummy,
        i32_dummy,
        i32_dummy,
        i32_dummy,
        i32_dummy,
        status,
    )
    scalars = (
        int(total_q),
        int(num_q_heads),
        int(num_kv_heads),
        topk,
        max_splits,
        max_task_claims,
        softmax_scale_log2,
        attention_mode,
        int(causal),
        int(derive_q_offset),
        seqlen_q,
        max_pages,
        split_policy,
    )
    signature = _launch_signature(
        variant=variant,
        target=target,
        tensors=tensors,
        scalars=scalars,
        grid=grid,
    )
    _check_warmed_launch(workspace, signature, capturing=capturing)
    _get_module(variant, target).run(
        *tensors,
        *scalars,
        *grid,
        stream_ptr,
    )
    _record_successful_launch(workspace, signature, capturing=capturing)


def blackwell_msa_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
    page_table: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    return_softmax_lse: bool = False,
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
    k_global_scale: Optional[float] = None,
    v_global_scale: Optional[float] = None,
    q_offset=None,
    return_temperature_lse: bool = False,
    lse_temperature_scale: float = 1.0,
    workspace: Optional[MSASparseAttentionWorkspace] = None,
):
    """Run sparse prefill on compute capability 10.0 or 10.3."""

    _validate_scale_arguments(
        q=q,
        k=k,
        v=v,
        k_scale=k_scale,
        v_scale=v_scale,
        k_global_scale=k_global_scale,
        v_global_scale=v_global_scale,
        allow_uniform_fp8=False,
    )
    total_q, num_q_heads, num_kv_heads, group_size = _validate_attention_tensors(
        q, k, v, q2k_indices
    )
    if q.dtype == torch.float8_e4m3fn:
        raise NotImplementedError(
            "uniform FP8 Q/K/V is supported only by sparse decode"
        )
    capturing = torch.cuda.is_current_stream_capturing()
    if capturing and workspace is None:
        raise RuntimeError(
            "CUDA graph capture of MSA on compute capability 10.0/10.3 "
            "requires an explicit MSASparseAttentionWorkspace warmed with the "
            "exact tensors and capture stream"
        )
    if workspace is not None and not isinstance(workspace, MSASparseAttentionWorkspace):
        raise TypeError("workspace must be an MSASparseAttentionWorkspace")
    stream_ptr = _stream_ptr(q.device)
    context = workspace._lock if workspace is not None else nullcontext()
    with context:
        if workspace is not None:
            _bind_workspace(
                workspace,
                device=q.device,
                stream_ptr=stream_ptr,
                capturing=capturing,
            )
        cu_q = _require_cuda_i32(
            cu_seqlens_q,
            device=q.device,
            name="cu_seqlens_q",
        )
        batch_size = cu_q.numel() - 1
        if batch_size <= 0:
            raise ValueError("cu_seqlens_q must contain at least two entries")
        paged, cu_k, kv_lens, page_table_arg, max_pages = _prepare_layout(
            q=q,
            k=k,
            page_table=page_table,
            seqused_k=seqused_k,
            cu_seqlens_k=cu_seqlens_k,
            batch_size=batch_size,
            prefill=True,
            workspace=workspace,
        )
        derive_q_offset = q_offset is None
        q_offsets = (
            cu_k
            if derive_q_offset
            else _explicit_q_offsets(
                q_offset,
                batch_size=batch_size,
                device=q.device,
                workspace=workspace,
                name="prefill_q_offsets",
            )
        )
        scale = _HEAD_DIM**-0.5 if softmax_scale is None else float(softmax_scale)
        if not math.isfinite(scale):
            raise ValueError("softmax_scale must be finite")
        temperature_scale = float(lse_temperature_scale)
        if not math.isfinite(temperature_scale) or temperature_scale <= 0:
            raise ValueError("lse_temperature_scale must be positive and finite")
        target = _select_target(q.device)
        requested_prefill_schedule = os.environ.get(
            "FLASHINFER_MSA_PREFILL_SCHEDULE", ""
        )
        if requested_prefill_schedule not in {"", "m64"}:
            raise ValueError("FLASHINFER_MSA_PREFILL_SCHEDULE must be empty or 'm64'")
        use_topk8_qagg = _is_exact_fp8_topk8_qagg_prefill(
            q=q,
            k=k,
            v=v,
            q2k_indices=q2k_indices,
            cu_q=cu_q,
            cu_k=cu_k,
            paged=paged,
            batch_size=batch_size,
            causal=causal,
            q_offset_is_none=q_offset is None,
            softmax_scale=softmax_scale,
            return_temperature_lse=return_temperature_lse,
            lse_temperature_scale=temperature_scale,
            requested_schedule=requested_prefill_schedule,
            capturing=capturing,
        )
        use_topk4_qload4 = _is_exact_bf16_topk4_qload4_prefill(
            q=q,
            k=k,
            v=v,
            q2k_indices=q2k_indices,
            cu_q=cu_q,
            cu_k=cu_k,
            page_table=page_table_arg,
            kv_lens=kv_lens,
            paged=paged,
            batch_size=batch_size,
            causal=causal,
            q_offset_is_none=q_offset is None,
            softmax_scale=softmax_scale,
            return_temperature_lse=return_temperature_lse,
            lse_temperature_scale=temperature_scale,
            requested_schedule=requested_prefill_schedule,
        )
        topk = int(q2k_indices.shape[2])
        if topk != _ATTENTION_TOPK and not (use_topk8_qagg or use_topk4_qload4):
            raise ValueError(
                "non-TopK16 Blackwell MSA attention is restricted to exact routes"
            )
        if use_topk8_qagg:
            exact_out, exact_lse, exact_temperature_lse = (
                _run_exact_fp8_topk8_qagg_prefill(
                    target=target,
                    q=q,
                    k=k,
                    v=v,
                    q2k_indices=q2k_indices,
                    cu_q=cu_q,
                    cu_k=cu_k,
                    stream_ptr=stream_ptr,
                    workspace=workspace,
                )
            )
            if return_temperature_lse:
                return exact_out, exact_lse, exact_temperature_lse
            if return_softmax_lse:
                return exact_out, exact_lse
            return exact_out
        if use_topk4_qload4:
            exact_out, exact_lse, exact_temperature_lse = (
                _run_exact_bf16_topk4_qload4_prefill(
                    target=target,
                    q=q,
                    k=k,
                    v=v,
                    q2k_indices=q2k_indices,
                    cu_q=cu_q,
                    cu_k=cu_k,
                    kv_lens=kv_lens,
                    page_table=page_table_arg,
                    return_softmax_lse=return_softmax_lse,
                    return_temperature_lse=return_temperature_lse,
                    stream_ptr=stream_ptr,
                    workspace=workspace,
                    capturing=capturing,
                )
            )
            if return_temperature_lse:
                return exact_out, exact_lse, exact_temperature_lse
            if return_softmax_lse:
                return exact_out, exact_lse
            return exact_out

        out = _workspace_buffer(
            workspace,
            "prefill_out",
            tuple(q.shape),
            dtype=q.dtype,
            device=q.device,
        )
        lse = _workspace_buffer(
            workspace,
            "prefill_lse",
            (total_q, num_q_heads),
            dtype=torch.float32,
            device=q.device,
        )
        temperature_lse = _workspace_buffer(
            workspace,
            "prefill_temperature_lse",
            (total_q, num_q_heads),
            dtype=torch.float32,
            device=q.device,
        )
        if _should_use_long_prefill(
            requested_schedule=requested_prefill_schedule,
            batch_size=batch_size,
            total_q=total_q,
            paged=paged,
            group_size=group_size,
            max_pages=max_pages,
            k_outer_dim=int(k.shape[0]),
            q_dtype=q.dtype,
            k_dtype=k.dtype,
            v_dtype=v.dtype,
            causal=causal,
            q_offset_is_none=q_offset is None,
            return_temperature_lse=return_temperature_lse,
            lse_temperature_scale=temperature_scale,
        ):
            _run_long_prefill_modules(
                target=target,
                q=q,
                k=k,
                v=v,
                out=out,
                lse=lse,
                temperature_lse=temperature_lse,
                q2k_indices=q2k_indices,
                cu_q=cu_q,
                cu_k=cu_k,
                q_offsets=q_offsets,
                kv_lens=kv_lens,
                page_table=page_table_arg,
                paged=paged,
                group_size=group_size,
                max_pages=max_pages,
                softmax_scale_log2=scale / math.log(2.0),
                lse_temperature_scale=temperature_scale,
                return_softmax_lse=return_softmax_lse,
                return_temperature_lse=return_temperature_lse,
                stream_ptr=stream_ptr,
                workspace=workspace,
                capturing=capturing,
            )
            if return_temperature_lse:
                return out, lse, temperature_lse
            if return_softmax_lse:
                return out, lse
            return out
        folded_gqa_group = (
            group_size
            if group_size in {8, 16}
            and q.dtype == torch.bfloat16
            and k.dtype == torch.bfloat16
            else 0
        )
        max_kv_len = 0
        if requested_prefill_schedule == "m64" and not paged:
            max_kv_len = (
                int(k.shape[0])
                if batch_size == 1
                else int((cu_k[1:] - cu_k[:-1]).max().item())
            )
        use_m64_exact_union = (
            requested_prefill_schedule == "m64"
            and folded_gqa_group == 16
            and not paged
            and not capturing
            and max_kv_len <= 64 * _BLOCK_SIZE
        )
        if use_m64_exact_union:
            variant: BlackwellMSAVariant = "prefill_m64_bf16_gqa16_flat"
            q_tile = _M64_GQA16_Q_TILE
        else:
            variant = _prefill_variant(
                q_dtype=q.dtype,
                k_dtype=k.dtype,
                paged=paged,
                folded_gqa_group=folded_gqa_group,
                causal=causal,
                max_pages=max_pages,
            )
            q_tile = (
                _M128_GQA8_Q_TILE
                if folded_gqa_group == 8
                else (_M128_GQA16_Q_TILE if folded_gqa_group == 16 else _M128_Q_TILE)
            )
        grid = (
            (total_q + q_tile - 1) // q_tile + batch_size - 1,
            num_kv_heads if folded_gqa_group else num_q_heads,
            1,
        )
        _run_prefill_module(
            variant=variant,
            target=target,
            q=q,
            k=k,
            v=v,
            out=out,
            lse=lse,
            temperature_lse=temperature_lse,
            q2k_indices=q2k_indices,
            cu_q=cu_q,
            cu_k=cu_k,
            q_offsets=q_offsets,
            kv_lens=kv_lens,
            page_table=page_table_arg,
            total_q=total_q,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            topk=int(q2k_indices.shape[2]),
            batch_size=batch_size,
            uniform_q_len=0,
            max_pages=max_pages,
            causal=causal,
            derive_q_offset=derive_q_offset,
            softmax_scale_log2=scale / math.log(2.0),
            lse_temperature_scale=temperature_scale,
            return_softmax_lse=return_softmax_lse,
            return_temperature_lse=return_temperature_lse,
            grid=grid,
            stream_ptr=stream_ptr,
            workspace=workspace,
            capturing=capturing,
        )
    if return_temperature_lse:
        return out, lse, temperature_lse
    if return_softmax_lse:
        return out, lse
    return out


def blackwell_msa_sparse_decode_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_indices: torch.Tensor,
    *,
    page_table: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqlen_q: int = 1,
    causal: bool = True,
    softmax_scale: Optional[float] = None,
    return_softmax_lse: bool = False,
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
    k_global_scale: Optional[float] = None,
    v_global_scale: Optional[float] = None,
    q_offset=None,
    partial_dtype: Optional[torch.dtype] = None,
    force_fused: Optional[bool] = None,
    workspace: Optional[MSASparseAttentionWorkspace] = None,
):
    """Run sparse decode on compute capability 10.0 or 10.3."""

    del partial_dtype
    k_global_multiplier, output_scale = _validate_scale_arguments(
        q=q,
        k=k,
        v=v,
        k_scale=k_scale,
        v_scale=v_scale,
        k_global_scale=k_global_scale,
        v_global_scale=v_global_scale,
        allow_uniform_fp8=True,
    )
    total_q, num_q_heads, num_kv_heads, _ = _validate_attention_tensors(
        q, k, v, q2k_indices
    )
    if seqlen_q <= 0 or total_q % seqlen_q:
        raise ValueError("q rows must equal batch_size * positive seqlen_q")
    if force_fused not in (None, True, False):
        raise ValueError("force_fused must be True, False, or None")
    batch_size = total_q // seqlen_q
    capturing = torch.cuda.is_current_stream_capturing()
    if capturing and workspace is None:
        raise RuntimeError(
            "CUDA graph capture of MSA on compute capability 10.0/10.3 "
            "requires an explicit MSASparseAttentionWorkspace warmed with the "
            "exact tensors and capture stream"
        )
    if workspace is not None and not isinstance(workspace, MSASparseAttentionWorkspace):
        raise TypeError("workspace must be an MSASparseAttentionWorkspace")
    stream_ptr = _stream_ptr(q.device)
    context = workspace._lock if workspace is not None else nullcontext()
    with context:
        if workspace is not None:
            _bind_workspace(
                workspace,
                device=q.device,
                stream_ptr=stream_ptr,
                capturing=capturing,
            )
        paged, cu_k, kv_lens, page_table_arg, max_pages = _prepare_layout(
            q=q,
            k=k,
            page_table=page_table,
            seqused_k=seqused_k,
            cu_seqlens_k=cu_seqlens_k,
            batch_size=batch_size,
            prefill=False,
            workspace=workspace,
        )
        cu_q = q2k_indices.reshape(-1)
        derive_q_offset = q_offset is None
        if derive_q_offset:
            q_offsets = cu_k
        else:
            q_offsets = _explicit_q_offsets(
                q_offset,
                batch_size=batch_size,
                device=q.device,
                workspace=workspace,
                name="decode_explicit_q_offsets",
            )
        scale = _HEAD_DIM**-0.5 if softmax_scale is None else float(softmax_scale)
        scale *= k_global_multiplier
        if not math.isfinite(scale):
            raise ValueError("softmax_scale must be finite")
        if not math.isfinite(output_scale):
            raise ValueError("v_global_scale must be finite")
        target = _select_target(q.device)
        requested_schedule = os.environ.get("FLASHINFER_MSA_FP8_Q1_SCHEDULE", "")
        valid_schedules = {
            "",
            "batch_attention",
            "q1_exact",
            "q1_flat_xform2",
            "q1_paged_xform2",
            "paged_uniform_fp8",
        }
        if requested_schedule not in valid_schedules:
            raise ValueError(
                "FLASHINFER_MSA_FP8_Q1_SCHEDULE must be batch_attention, "
                "q1_exact, q1_flat_xform2, q1_paged_xform2, or paged_uniform_fp8"
            )
        non16_variant = _exact_non16_decode_variant(
            requested_schedule=requested_schedule,
            capturing=capturing,
            paged=paged,
            force_fused=force_fused,
            causal=causal,
            q_offset_is_none=q_offset is None,
            q_dtype=q.dtype,
            k_dtype=k.dtype,
            batch_size=batch_size,
            total_q=total_q,
            seqlen_q=seqlen_q,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            topk=int(q2k_indices.shape[2]),
            k_outer_dim=int(k.shape[0]),
            max_pages=max_pages,
        )
        if int(q2k_indices.shape[2]) != _ATTENTION_TOPK and non16_variant is None:
            raise ValueError(
                "non-TopK16 Blackwell MSA attention is restricted to exact routes"
            )
        out = _workspace_buffer(
            workspace,
            "decode_out",
            tuple(q.shape),
            dtype=(torch.bfloat16 if q.dtype == torch.float8_e4m3fn else q.dtype),
            device=q.device,
        )
        lse = _workspace_buffer(
            workspace,
            "decode_lse",
            (total_q, num_q_heads),
            dtype=torch.float32,
            device=q.device,
        )
        if non16_variant is not None:
            _run_decode_module(
                variant=non16_variant,
                target=target,
                q=q,
                k=k,
                v=v,
                out=out,
                lse=lse,
                q2k_indices=q2k_indices,
                cu_q=cu_q,
                cu_k=cu_k,
                q_offsets=q_offsets,
                kv_lens=kv_lens,
                page_table=page_table_arg,
                topk=int(q2k_indices.shape[2]),
                max_pages=max_pages,
                seqlen_q=seqlen_q,
                softmax_scale_log2=scale / math.log(2.0),
                causal=causal,
                paged=paged,
                derive_q_offset=derive_q_offset,
                workspace=workspace,
                capturing=capturing,
                stream_ptr=stream_ptr,
            )
            return (out, lse) if return_softmax_lse else out
        uniform_fp8 = q.dtype == k.dtype == v.dtype == torch.float8_e4m3fn
        uniform_fp8_direct = (
            paged
            and force_fused is True
            and causal
            and q_offset is None
            and uniform_fp8
            and 1 <= seqlen_q <= 32
            and int(q2k_indices.shape[2]) == _ATTENTION_TOPK
        )
        if uniform_fp8:
            if requested_schedule not in {"", "paged_uniform_fp8"}:
                raise ValueError("uniform FP8 Q/K/V decode requires paged_uniform_fp8")
            if not uniform_fp8_direct:
                raise ValueError(
                    "uniform FP8 Q/K/V requires paged causal Q1-Q32/topk16, "
                    "force_fused=True, and no explicit q_offset"
                )
            fp8_schedule = "paged_uniform_fp8"
        else:
            fp8_schedule = _resolve_fp8_q1_schedule(
                requested=(
                    ""
                    if requested_schedule == "batch_attention"
                    else requested_schedule
                ),
                capturing=capturing,
                paged=paged,
                force_fused=force_fused,
                causal=causal,
                q_offset_is_none=q_offset is None,
                q_dtype=q.dtype,
                k_dtype=k.dtype,
                batch_size=batch_size,
                total_q=total_q,
                seqlen_q=seqlen_q,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                topk=int(q2k_indices.shape[2]),
                k_outer_dim=int(k.shape[0]),
                max_pages=max_pages,
            )
        if fp8_schedule in {"q1_exact", "q1_flat_xform2", "q1_paged_xform2"}:
            common = (
                not capturing
                and seqlen_q == 1
                and force_fused is True
                and q.dtype == torch.bfloat16
                and k.dtype == torch.float8_e4m3fn
                and num_q_heads == 64
                and causal
                and q_offset is None
            )
            if fp8_schedule == "q1_exact":
                expected_batch = 128 if paged else 32
                expected_outer = expected_batch * (32 if paged else 4096)
                valid_direct_shape = (
                    common
                    and batch_size == expected_batch
                    and total_q == expected_batch
                    and num_kv_heads == 8
                    and int(k.shape[0]) == expected_outer
                    and (not paged or max_pages == 32)
                )
            elif fp8_schedule == "q1_flat_xform2":
                valid_direct_shape = (
                    common
                    and not paged
                    and batch_size == total_q == 32
                    and num_kv_heads == 4
                    and int(k.shape[0]) == 32 * 8192
                )
            else:
                valid_direct_shape = (
                    common
                    and paged
                    and batch_size == total_q == 128
                    and num_kv_heads == 4
                    and int(k.shape[0]) == 128 * 32
                    and max_pages == 32
                )
            if not valid_direct_shape:
                raise ValueError(
                    f"{fp8_schedule} is restricted to its frozen serving shape; "
                    "causal, force_fused=True, no explicit q_offset, and eager execution are required"
                )
        if fp8_schedule in {
            "q1_exact",
            "q1_flat_xform2",
            "q1_paged_xform2",
            "paged_uniform_fp8",
        }:
            _run_fp8_direct_module(
                schedule=fp8_schedule,
                target=target,
                q=q,
                k=k,
                v=v,
                out=out,
                lse=lse,
                q2k_indices=q2k_indices,
                cu_k=cu_k,
                q_offsets=q_offsets,
                kv_lens=kv_lens,
                page_table=page_table_arg,
                paged=paged,
                max_pages=max_pages,
                seqlen_q=seqlen_q,
                softmax_scale_log2=scale / math.log(2.0),
                output_scale=output_scale,
                stream_ptr=stream_ptr,
                workspace=workspace,
                capturing=capturing,
            )
            return (out, lse) if return_softmax_lse else out
        variant = _decode_variant(
            q_dtype=q.dtype,
            k_dtype=k.dtype,
            paged=paged,
        )
        _run_decode_module(
            variant=variant,
            target=target,
            q=q,
            k=k,
            v=v,
            out=out,
            lse=lse,
            q2k_indices=q2k_indices,
            cu_q=cu_q,
            cu_k=cu_k,
            q_offsets=q_offsets,
            kv_lens=kv_lens,
            page_table=page_table_arg,
            topk=int(q2k_indices.shape[2]),
            max_pages=max_pages,
            seqlen_q=seqlen_q,
            softmax_scale_log2=scale / math.log(2.0),
            causal=causal,
            paged=paged,
            derive_q_offset=derive_q_offset,
            workspace=workspace,
            capturing=capturing,
            stream_ptr=stream_ptr,
        )
    return (out, lse) if return_softmax_lse else out


def blackwell_msa_topk_select(
    max_score: torch.Tensor,
    topk: int,
    num_valid_pages: Optional[int] = None,
    output: Optional[torch.Tensor] = None,
    force_begin_blocks: int = 0,
    force_end_blocks: int = 0,
) -> torch.Tensor:
    """Select exact top-16 block indices on compute capability 10.0/10.3."""

    if not isinstance(max_score, torch.Tensor) or not max_score.is_cuda:
        raise ValueError("max_score must be a CUDA tensor")
    if max_score.dtype != torch.float32:
        raise ValueError(f"max_score must be float32, got {max_score.dtype}")
    if max_score.ndim != 3 or not max_score.is_contiguous():
        raise ValueError(
            "max_score must be contiguous with shape "
            "(num_q_heads, max_k_tiles, total_q)"
        )
    if topk != _TOPK_SELECT:
        raise ValueError(f"topk must be {_TOPK_SELECT}, got {topk}")
    num_heads, max_k_tiles, total_q = (int(value) for value in max_score.shape)
    if min(num_heads, max_k_tiles, total_q) <= 0:
        raise ValueError("max_score dimensions must be positive")
    valid = max_k_tiles if num_valid_pages is None else int(num_valid_pages)
    if not 0 < valid <= max_k_tiles:
        raise ValueError(f"num_valid_pages must be in (0, {max_k_tiles}], got {valid}")
    forced = force_begin_blocks + force_end_blocks
    if force_begin_blocks < 0 or force_end_blocks < 0:
        raise ValueError("force_begin_blocks and force_end_blocks must be non-negative")
    if forced > topk or forced > valid:
        raise ValueError(
            "force_begin_blocks + force_end_blocks must not exceed topk or "
            "num_valid_pages"
        )
    expected_shape = (total_q, num_heads, topk)
    if output is None:
        output = torch.empty(
            expected_shape,
            dtype=torch.int32,
            device=max_score.device,
        )
    elif (
        output.device != max_score.device
        or output.dtype != torch.int32
        or tuple(output.shape) != expected_shape
        or not output.is_contiguous()
    ):
        raise ValueError(
            f"output must be contiguous CUDA int32 with shape {expected_shape}"
        )
    target = _select_target(max_score.device)
    device_index = (
        max_score.device.index
        if max_score.device.index is not None
        else torch.cuda.current_device()
    )
    warm_key = (device_index, target)
    capturing = torch.cuda.is_current_stream_capturing()
    with _topk_warmed_devices_lock:
        warmed = warm_key in _topk_warmed_devices
    if capturing and not warmed:
        raise RuntimeError(
            "msa_topk_select must be invoked eagerly on this device before "
            "CUDA graph capture"
        )
    stream_ptr = _stream_ptr(max_score.device)
    _get_module("topk", target).run(
        max_score,
        output,
        num_heads,
        max_k_tiles,
        total_q,
        valid,
        int(force_begin_blocks),
        int(force_end_blocks),
        total_q * num_heads,
        1,
        1,
        stream_ptr,
    )
    if not capturing:
        with _topk_warmed_devices_lock:
            _topk_warmed_devices.add(warm_key)
    return output


__all__ = [
    "MSASparseAttentionWorkspace",
    "blackwell_msa_sparse_attention",
    "blackwell_msa_sparse_decode_attention",
    "blackwell_msa_topk_select",
    "is_blackwell_msa_device",
]
