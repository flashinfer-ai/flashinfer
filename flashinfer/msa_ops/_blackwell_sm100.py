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
import threading
import weakref
from contextlib import nullcontext
from typing import TYPE_CHECKING, Optional

import torch

from ..utils import get_compute_capability

if TYPE_CHECKING:
    from ..jit.blackwell_msa import BlackwellMSATarget, BlackwellMSAVariant

_BLOCK_SIZE = 128
_HEAD_DIM = 128
_TOPK_SELECT = 16
_SUPPORTED_ATTENTION_TOPK = {4, 8, 16, 32}
_SUPPORTED_COMPUTE_CAPABILITIES = {(10, 0), (10, 3)}
_M64_Q_TILE = 8
_M128_Q_TILE = 256
_M128_GQA16_Q_TILE = 16

_MODE_DECODE_ONLY = 1
_MODE_MIXED = 3
_SPLIT_ADAPTIVE = 0
_SPLIT_FORCED = 1


class MSASparseAttentionWorkspace:
    """Caller-owned storage for CC10 MSA CUDA graph capture.

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
        self._routes: dict[tuple, tuple] = {}
        self._warmed_launches: set[tuple] = set()
        self._bound_stream_ptr: Optional[int] = None
        self._captured = False


_topk_warmed_devices: set[tuple[int, str]] = set()
_topk_warmed_devices_lock = threading.Lock()
_eager_decode_dummies: dict[tuple[int, int, torch.dtype], torch.Tensor] = {}
_eager_decode_dummies_lock = threading.Lock()
_flat_kv_route_cache: dict[
    int, tuple[weakref.ReferenceType[torch.Tensor], int, int, bool]
] = {}
_flat_kv_route_cache_lock = threading.Lock()


def is_blackwell_msa_device(device: torch.device | str) -> bool:
    """Return whether ``device`` is a supported CC10 MSA target."""

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
            "the CC10 MSA backend requires compute capability 10.0 or 10.3; "
            f"got {compute_capability[0]}.{compute_capability[1]}"
        )
    if _cuda_version_at_least("12.9"):
        return "sm100f"
    if compute_capability == (10, 0) and _cuda_version_at_least("12.8"):
        return "sm100a"
    if compute_capability == (10, 3):
        raise RuntimeError("MSA on compute capability 10.3 requires CUDA 12.9 or newer")
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


def _uniform_cu_seqlens_q(
    *,
    batch_size: int,
    seqlen_q: int,
    device: torch.device,
    workspace: Optional[MSASparseAttentionWorkspace],
) -> torch.Tensor:
    cu_q = _workspace_buffer(
        workspace,
        "decode_cu_seqlens_q",
        (batch_size + 1,),
        dtype=torch.int32,
        device=device,
    )
    torch.arange(
        0,
        (batch_size + 1) * seqlen_q,
        seqlen_q,
        dtype=torch.int32,
        device=device,
        out=cu_q,
    )
    return cu_q


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


def _flat_kv_lengths(
    cu_k: torch.Tensor,
    *,
    workspace: Optional[MSASparseAttentionWorkspace],
) -> torch.Tensor:
    kv_lens = _workspace_buffer(
        workspace,
        "decode_kv_lens",
        (cu_k.numel() - 1,),
        dtype=torch.int32,
        device=cu_k.device,
    )
    torch.sub(cu_k[1:], cu_k[:-1], out=kv_lens)
    return kv_lens


def _decode_default_offsets(
    kv_lens: torch.Tensor,
    *,
    seqlen_q: int,
    workspace: Optional[MSASparseAttentionWorkspace],
) -> torch.Tensor:
    offsets = _workspace_buffer(
        workspace,
        "decode_q_offsets",
        (kv_lens.numel(),),
        dtype=torch.int32,
        device=kv_lens.device,
    )
    offsets.copy_(kv_lens)
    offsets.sub_(seqlen_q)
    return offsets


def _flat_kv_route_properties(
    cu_k: torch.Tensor, kv_lens: torch.Tensor
) -> tuple[int, bool]:
    """Resolve and cache the properties used by the eager M64 route."""

    def resolve() -> tuple[int, bool]:
        lengths = kv_lens.cpu().tolist()
        return (
            max(lengths, default=0),
            all(length % _BLOCK_SIZE == 0 for length in lengths),
        )

    tensor_id = id(cu_k)
    try:
        version = cu_k._version
    except RuntimeError:
        # Tensors created in inference mode do not expose a version counter,
        # so their contents cannot be cached safely.
        return resolve()
    with _flat_kv_route_cache_lock:
        cached = _flat_kv_route_cache.get(tensor_id)
        if cached is not None and cached[0]() is cu_k and cached[1] == version:
            return cached[2], cached[3]
    maximum, block_aligned = resolve()
    with _flat_kv_route_cache_lock:
        if len(_flat_kv_route_cache) >= 64:
            dead_keys = [
                key for key, value in _flat_kv_route_cache.items() if value[0]() is None
            ]
            for key in dead_keys:
                _flat_kv_route_cache.pop(key, None)
            if len(_flat_kv_route_cache) >= 64:
                _flat_kv_route_cache.clear()
        _flat_kv_route_cache[tensor_id] = (
            weakref.ref(cu_k),
            version,
            maximum,
            block_aligned,
        )
    return maximum, block_aligned


def _validate_scale_arguments(
    *,
    k: torch.Tensor,
    k_scale,
    v_scale,
    k_global_scale,
    v_global_scale,
) -> None:
    if k.dtype == torch.uint8:
        raise NotImplementedError(
            "NVFP4 K/V is not supported by MSA on compute capability 10.0/10.3"
        )
    if any(
        value is not None
        for value in (k_scale, v_scale, k_global_scale, v_global_scale)
    ):
        raise NotImplementedError(
            "K/V scale arguments are not supported by MSA on compute "
            "capability 10.0/10.3"
        )


def _validate_attention_tensors(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_indices: torch.Tensor,
) -> tuple[int, int, int, int]:
    if not isinstance(q, torch.Tensor) or not q.is_cuda:
        raise ValueError("q must be a CUDA tensor")
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"q must be bf16/fp16, got {q.dtype}")
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
        if q.dtype != torch.bfloat16:
            raise NotImplementedError(
                "FP8 K/V with FP16 Q is not supported on compute capability 10.0/10.3"
            )
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
        raise ValueError("attention topk must be one of 4, 8, 16, or 32")
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
    single_token_decode: bool = False,
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
    kv_lens = (
        cu_k
        if prefill or single_token_decode
        else _flat_kv_lengths(cu_k, workspace=workspace)
    )
    return False, cu_k, kv_lens, q.reshape(-1).view(torch.int32), 0


def _prefill_variant(
    *,
    q_dtype: torch.dtype,
    k_dtype: torch.dtype,
    paged: bool,
    folded_gqa16: bool,
) -> "BlackwellMSAVariant":
    layout = "paged" if paged else "flat"
    if folded_gqa16:
        return f"prefill_m128_bf16_gqa16_{layout}"  # type: ignore[return-value]
    if k_dtype == torch.float8_e4m3fn:
        return f"prefill_m128_fp8_{layout}"  # type: ignore[return-value]
    dtype_name = "fp16" if q_dtype == torch.float16 else "bf16"
    return f"prefill_m128_{dtype_name}_{layout}"  # type: ignore[return-value]


def _decode_variant(
    *,
    q_dtype: torch.dtype,
    k_dtype: torch.dtype,
    paged: bool,
    specialized_m16: bool,
) -> "BlackwellMSAVariant":
    layout = "paged" if paged else "flat"
    if specialized_m16:
        return f"decode_m16_bf16_{layout}"  # type: ignore[return-value]
    if k_dtype == torch.float8_e4m3fn:
        return f"decode_fp8_{layout}"  # type: ignore[return-value]
    dtype_name = "fp16" if q_dtype == torch.float16 else "bf16"
    return f"decode_{dtype_name}_{layout}"  # type: ignore[return-value]


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
        int(causal),
        int(derive_q_offset),
        softmax_scale_log2,
        lse_temperature_scale,
        int(return_softmax_lse or return_temperature_lse),
        int(return_temperature_lse),
    )
    if variant == "prefill_m64_bf16_flat":
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
    force_fused: Optional[bool],
    persistent_unsplit: bool,
    derive_q_offset: bool,
    workspace: Optional[MSASparseAttentionWorkspace],
    capturing: bool,
    stream_ptr: int,
) -> None:
    fp8 = k.dtype == torch.float8_e4m3fn
    i32_dummy = q2k_indices.reshape(-1)
    f32_dummy = lse.reshape(-1)
    q_prefill_dummy = _decode_tma_dummy(
        device=q.device,
        stream_ptr=stream_ptr,
        dtype=q.dtype,
        workspace=workspace,
    )
    k_pair_dummy = q_prefill_dummy.reshape(2, 1, 128, _HEAD_DIM)
    v_pair_dummy = k_pair_dummy
    if fp8:
        k_launch = k.view(torch.uint8)
        v_launch = v.view(torch.uint8)
    else:
        k_launch = k
        v_launch = v
    page_table_arg = page_table if paged else i32_dummy
    total_q, num_q_heads, _ = q.shape
    num_kv_heads = k.shape[1]

    if force_fused is True:
        grid = (int(total_q), int(num_kv_heads), 1)
        qo_indptr = cu_q
        partial_o = f32_dummy
        partial_m = f32_dummy
        partial_d = f32_dummy
        split_completion = i32_dummy
        status = i32_dummy
        max_splits = 1
        max_task_claims = 1
        attention_mode = _MODE_DECODE_ONLY
        split_policy = _SPLIT_ADAPTIVE
    else:
        num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count
        max_splits = 1 if persistent_unsplit else topk // 2
        max_task_claims = int(total_q) * int(num_kv_heads) * max_splits
        grid = (min(num_sms, max_task_claims), 1, 1)
        qo_indptr = i32_dummy
        if persistent_unsplit:
            partial_o = f32_dummy
            partial_m = f32_dummy
            partial_d = f32_dummy
            split_completion = i32_dummy
        else:
            partial_slots = int(total_q) * int(num_kv_heads) * max_splits
            partial_o = _workspace_buffer(
                workspace,
                "decode_partial_o",
                (partial_slots, _BLOCK_SIZE, _HEAD_DIM),
                dtype=torch.float32,
                device=q.device,
            )
            partial_m = _workspace_buffer(
                workspace,
                "decode_partial_m",
                (partial_slots, _BLOCK_SIZE),
                dtype=torch.float32,
                device=q.device,
            )
            partial_d = _workspace_buffer(
                workspace,
                "decode_partial_d",
                (partial_slots, _BLOCK_SIZE),
                dtype=torch.float32,
                device=q.device,
            )
            split_completion = _workspace_buffer(
                workspace,
                "decode_split_completion",
                (int(total_q) * int(num_kv_heads),),
                dtype=torch.int32,
                device=q.device,
                zero=True,
            )
        status = _workspace_buffer(
            workspace,
            "decode_status",
            (2,),
            dtype=torch.int32,
            device=q.device,
            zero=True,
        )
        attention_mode = _MODE_DECODE_ONLY if persistent_unsplit else _MODE_MIXED
        split_policy = _SPLIT_FORCED if force_fused is False else _SPLIT_ADAPTIVE

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


def _route_key(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_indices: torch.Tensor,
    page_table: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    seqused_k: Optional[torch.Tensor],
    seqlen_q: int,
    force_fused: Optional[bool],
) -> tuple:
    optional_tensors = tuple(
        None if tensor is None else _tensor_signature(tensor)
        for tensor in (page_table, cu_seqlens_k, seqused_k)
    )
    return (
        "decode_route",
        *(_tensor_signature(tensor) for tensor in (q, k, v, q2k_indices)),
        *optional_tensors,
        seqlen_q,
        force_fused,
    )


def _is_long_paged_gqa16_direct_decode(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    group_size: int,
    seqlen_q: int,
    paged: bool,
    force_fused: Optional[bool],
) -> bool:
    """Match the long Q8 paged geometry where selected-block decode is cheaper."""

    return (
        paged
        and seqlen_q == 8
        and force_fused is True
        and group_size == 16
        and q.dtype == torch.bfloat16
        and k.dtype == torch.bfloat16
        and tuple(q.shape) == (512, 64, _HEAD_DIM)
        and tuple(k.shape) == (32768, 4, _BLOCK_SIZE, _HEAD_DIM)
    )


def _select_decode_route(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    cu_k: torch.Tensor,
    kv_lens: torch.Tensor,
    group_size: int,
    seqlen_q: int,
    paged: bool,
    force_fused: Optional[bool],
    workspace: Optional[MSASparseAttentionWorkspace],
    route_key: tuple,
    capturing: bool,
) -> tuple[str, bool, Optional[bool]]:
    if capturing:
        if workspace is None or route_key not in workspace._routes:
            raise RuntimeError(
                "MSA CUDA graph capture requires an eager warmup with the exact "
                "decode tensors and options"
            )
        return workspace._routes[route_key]  # type: ignore[return-value]

    if _is_long_paged_gqa16_direct_decode(
        q=q,
        k=k,
        group_size=group_size,
        seqlen_q=seqlen_q,
        paged=paged,
        force_fused=force_fused,
    ):
        # The folded-M128 route scans the 512-block visible range.  At this
        # exact long-context geometry, visiting the 32 selected blocks with
        # the direct swap-AB body removes that union-scan overcompute.
        return "decode", False, True

    if seqlen_q > 1 and force_fused is not False:
        folded_gqa16 = (
            group_size == 16 and q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16
        )
        if folded_gqa16:
            use_m64 = False
            # M64 is the eager fast path for block-aligned KV sequences. A
            # graph workspace may be replayed after cu_k contents change, and
            # the flat allocation shape cannot prove per-sequence alignment,
            # so workspace-backed calls conservatively use M128.
            if not paged and seqlen_q <= _M64_Q_TILE and workspace is None:
                max_kv_len, block_aligned = _flat_kv_route_properties(cu_k, kv_lens)
                use_m64 = block_aligned and max_kv_len <= 64 * _BLOCK_SIZE
            if use_m64:
                return "m64", False, None
            return "m128", False, None

    use_m16 = seqlen_q == 1 and q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16
    persistent_unsplit = False
    path_force_fused = force_fused
    if seqlen_q == 1 and force_fused is None:
        num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count
        task_count = int(q.shape[0]) * int(k.shape[1])
        persistent_unsplit = task_count >= 4 * num_sms
        path_force_fused = not persistent_unsplit
    return "m16" if use_m16 else "decode", persistent_unsplit, path_force_fused


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
        k=k,
        k_scale=k_scale,
        v_scale=v_scale,
        k_global_scale=k_global_scale,
        v_global_scale=v_global_scale,
    )
    total_q, num_q_heads, num_kv_heads, group_size = _validate_attention_tensors(
        q, k, v, q2k_indices
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
            single_token_decode=False,
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
        folded_gqa16 = (
            group_size == 16 and q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16
        )
        variant = _prefill_variant(
            q_dtype=q.dtype,
            k_dtype=k.dtype,
            paged=paged,
            folded_gqa16=folded_gqa16,
        )
        q_tile = _M128_GQA16_Q_TILE if folded_gqa16 else _M128_Q_TILE
        grid = (
            (total_q + q_tile - 1) // q_tile + batch_size - 1,
            num_kv_heads if folded_gqa16 else num_q_heads,
            1,
        )
        target = _select_target(q.device)
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
    _validate_scale_arguments(
        k=k,
        k_scale=k_scale,
        v_scale=v_scale,
        k_global_scale=k_global_scale,
        v_global_scale=v_global_scale,
    )
    total_q, num_q_heads, num_kv_heads, group_size = _validate_attention_tensors(
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
            single_token_decode=seqlen_q == 1,
            workspace=workspace,
        )
        single_token = seqlen_q == 1
        cu_q = (
            q2k_indices.reshape(-1)
            if single_token
            else _uniform_cu_seqlens_q(
                batch_size=batch_size,
                seqlen_q=seqlen_q,
                device=q.device,
                workspace=workspace,
            )
        )
        derive_q_offset = single_token and q_offset is None
        if derive_q_offset:
            q_offsets = cu_k
        elif q_offset is None:
            q_offsets = _decode_default_offsets(
                kv_lens,
                seqlen_q=seqlen_q,
                workspace=workspace,
            )
        else:
            q_offsets = _explicit_q_offsets(
                q_offset,
                batch_size=batch_size,
                device=q.device,
                workspace=workspace,
                name="decode_explicit_q_offsets",
            )
        scale = _HEAD_DIM**-0.5 if softmax_scale is None else float(softmax_scale)
        if not math.isfinite(scale):
            raise ValueError("softmax_scale must be finite")
        out = _workspace_buffer(
            workspace,
            "decode_out",
            tuple(q.shape),
            dtype=q.dtype,
            device=q.device,
        )
        lse = _workspace_buffer(
            workspace,
            "decode_lse",
            (total_q, num_q_heads),
            dtype=torch.float32,
            device=q.device,
        )
        route_key = _route_key(
            q=q,
            k=k,
            v=v,
            q2k_indices=q2k_indices,
            page_table=page_table,
            cu_seqlens_k=cu_seqlens_k,
            seqused_k=seqused_k,
            seqlen_q=seqlen_q,
            force_fused=force_fused,
        )
        route, persistent_unsplit, path_force_fused = _select_decode_route(
            q=q,
            k=k,
            cu_k=cu_k,
            kv_lens=kv_lens,
            group_size=group_size,
            seqlen_q=seqlen_q,
            paged=paged,
            force_fused=force_fused,
            workspace=workspace,
            route_key=route_key,
            capturing=capturing,
        )
        target = _select_target(q.device)
        if route in {"m64", "m128"}:
            temperature_lse = _workspace_buffer(
                workspace,
                "decode_temperature_lse",
                (total_q, num_q_heads),
                dtype=torch.float32,
                device=q.device,
            )
            folded_gqa16 = route == "m128"
            if route == "m64":
                variant: BlackwellMSAVariant = "prefill_m64_bf16_flat"
                q_tile = _M64_Q_TILE
            else:
                variant = _prefill_variant(
                    q_dtype=q.dtype,
                    k_dtype=k.dtype,
                    paged=paged,
                    folded_gqa16=folded_gqa16,
                )
                q_tile = _M128_GQA16_Q_TILE
            grid = (
                (total_q + q_tile - 1) // q_tile + batch_size - 1,
                num_kv_heads,
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
                max_pages=max_pages,
                causal=causal,
                derive_q_offset=False,
                softmax_scale_log2=scale / math.log(2.0),
                lse_temperature_scale=1.0,
                return_softmax_lse=return_softmax_lse,
                return_temperature_lse=False,
                grid=grid,
                stream_ptr=stream_ptr,
                workspace=workspace,
                capturing=capturing,
            )
        else:
            specialized_m16 = route == "m16"
            variant = _decode_variant(
                q_dtype=q.dtype,
                k_dtype=k.dtype,
                paged=paged,
                specialized_m16=specialized_m16,
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
                force_fused=path_force_fused,
                persistent_unsplit=persistent_unsplit,
                derive_q_offset=derive_q_offset,
                workspace=workspace,
                capturing=capturing,
                stream_ptr=stream_ptr,
            )
        if workspace is not None and not capturing:
            workspace._routes[route_key] = (
                route,
                persistent_unsplit,
                path_force_fused,
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
