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

from dataclasses import dataclass
import functools as _functools
import math
import os
import warnings as _warnings
from typing import (
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    overload as _overload,
)

import torch

from ..api_logging import flashinfer_api
from ..autotuner import AutoTuner as _AutoTuner, TunableRunner as _TunableRunner
from ..jit import (
    gen_batch_mla_module as _gen_batch_mla_module,
    gen_trtllm_gen_fmha_module as _gen_trtllm_gen_fmha_module,
    setup_cubin_loader as _setup_cubin_loader,
)
from ..jit.mla import gen_mla_module as _gen_mla_module
from ..trace.templates.attention import (
    mla_paged_decode_trace as _mla_paged_decode_trace,
    trtllm_batch_decode_mla_trace_dispatch,
    xqa_batch_decode_mla_trace as _xqa_batch_decode_mla_trace,
)
from ..utils import (
    MaskMode as _MaskMode,
    check_shape_dtype_device,
    determine_mla_backend as _determine_mla_backend,
    device_support_pdl,
    get_compute_capability,
    get_device_sm_count,
    _get_trtllm_gen_multi_ctas_kv_counter_buffer,
    get_trtllm_gen_multi_ctas_kv_counter_bytes as _get_trtllm_gen_multi_ctas_kv_counter_bytes,
    is_sm12x_supported,
    log2e,
)
from ..xqa import xqa_mla as _xqa_mla
from ._sparse_mla_sm120 import _run_mla_decode_sparse_sm120


functools = _functools
warnings = _warnings
overload = _overload
AutoTuner = _AutoTuner
TunableRunner = _TunableRunner
gen_batch_mla_module = _gen_batch_mla_module
gen_trtllm_gen_fmha_module = _gen_trtllm_gen_fmha_module
setup_cubin_loader = _setup_cubin_loader
gen_mla_module = _gen_mla_module
mla_paged_decode_trace = _mla_paged_decode_trace
xqa_batch_decode_mla_trace = _xqa_batch_decode_mla_trace
MaskMode = _MaskMode
determine_mla_backend = _determine_mla_backend
get_trtllm_gen_multi_ctas_kv_counter_bytes = (
    _get_trtllm_gen_multi_ctas_kv_counter_bytes
)
xqa_mla = _xqa_mla


@dataclass(frozen=True)
class MLAHeadDimensions:
    """
    The dimensions of a single MLA head.

    Args:
        qk_nope_head_dim (int): The number of input channels without positional information in non-absorb mode.
        qk_rope_head_dim (int): The number of channels carrying positional information for both absorb and non-absorb modes.
        v_head_dim (int): The number of value channels, which is also the output head dimension in non-absorb mode.
        kv_lora_rank (int): The dimension of the compressed key-value representation across heads.
    """

    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    kv_lora_rank: int


deepseek_mla_dimensions = MLAHeadDimensions(
    qk_nope_head_dim=128,
    qk_rope_head_dim=64,
    v_head_dim=128,
    kv_lora_rank=512,
)

smaller_mla_dimensions = MLAHeadDimensions(
    qk_nope_head_dim=64,
    qk_rope_head_dim=64,
    v_head_dim=128,
    kv_lora_rank=256,
)

supported_mla_head_dimensions = [deepseek_mla_dimensions, smaller_mla_dimensions]


@dataclass(frozen=True)
class MLALayerDimensions:
    """
    The dimensions of an MLA layer.

    Args:
        head_dimensions (MLAHeadDimensions): The dimensions of a single MLA head.
        num_heads (int): The number of heads in the MLA layer.
    """

    head_dimensions: MLAHeadDimensions
    num_heads: int


supported_mla_layer_dimensions = [
    MLALayerDimensions(
        head_dimensions=deepseek_mla_dimensions, num_heads=128
    ),  # DSR1 dimensions
    MLALayerDimensions(
        head_dimensions=deepseek_mla_dimensions, num_heads=64
    ),  # GLM-5 dimensions
    MLALayerDimensions(
        head_dimensions=smaller_mla_dimensions, num_heads=32
    ),  # Smaller model dimensions
]


@dataclass(frozen=True)
class _SparseMLASegment:
    """Internal SM120 sparse MLA KV segment."""

    indices: torch.Tensor
    lengths: Optional[torch.Tensor] = None
    kv_cache: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class _NormalizedSparseMLASegment:
    kv_cache: Optional[torch.Tensor]
    indices: torch.Tensor
    lengths: Optional[torch.Tensor]


def _normalize_optional_mla_sink(
    sinks: Optional[Union[List[torch.Tensor], Tuple[torch.Tensor, ...], torch.Tensor]],
    backend_name: str,
) -> Optional[torch.Tensor]:
    if sinks is None:
        return None
    if isinstance(sinks, (list, tuple)):
        if len(sinks) != 1:
            raise ValueError(
                f"{backend_name} expects sinks to be a single tensor or a "
                f"length-1 list/tuple; got len={len(sinks)}."
            )
        return sinks[0]
    return sinks


def _normalize_sparse_mla_indices_and_lens(
    *,
    indices: torch.Tensor,
    lens: Optional[torch.Tensor],
    batch_size: int,
    q_len_per_request: int,
    device: torch.device,
    name: str,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if indices.ndim == 3:
        sparse_topk = int(indices.shape[-1])
        expected_shape: Tuple[int, ...] = (
            batch_size,
            q_len_per_request,
            sparse_topk,
        )
        if tuple(indices.shape) != expected_shape:
            raise ValueError(
                f"Expected {name}.shape == {expected_shape}, got {tuple(indices.shape)}"
            )
        indices = indices.reshape(batch_size * q_len_per_request, -1)
    elif indices.ndim == 2:
        sparse_topk = int(indices.shape[-1])
        expected_shape = (batch_size * q_len_per_request, sparse_topk)
        if tuple(indices.shape) != expected_shape:
            raise ValueError(
                f"Expected flattened {name}.shape == {expected_shape}, got "
                f"{tuple(indices.shape)}"
            )
    else:
        raise ValueError(f"{name} must have ndim 2 or 3, got {indices.ndim}")
    if sparse_topk <= 0:
        raise ValueError(f"{name} requires top-k > 0")
    if indices.dtype != torch.int32:
        raise ValueError(f"{name} must have dtype torch.int32, got {indices.dtype}")
    if indices.device != device:
        raise ValueError(f"{name} must be on device {device}, got {indices.device}")

    if lens is None:
        return indices, None
    if lens.dtype != torch.int32:
        raise ValueError(f"{name}_lens must have dtype torch.int32, got {lens.dtype}")
    if lens.device != device:
        raise ValueError(f"{name}_lens must be on device {device}, got {lens.device}")
    if lens.ndim == 2:
        expected_lens_shape: Tuple[int, ...] = (batch_size, q_len_per_request)
        if tuple(lens.shape) != expected_lens_shape:
            raise ValueError(
                f"Expected {name}_lens.shape == {expected_lens_shape}, got "
                f"{tuple(lens.shape)}"
            )
        lens = lens.reshape(-1)
    elif lens.ndim == 1:
        expected_lens_shape = (batch_size * q_len_per_request,)
        if tuple(lens.shape) != expected_lens_shape:
            raise ValueError(
                f"Expected flattened {name}_lens.shape == {expected_lens_shape}, "
                f"got {tuple(lens.shape)}"
            )
    else:
        raise ValueError(f"{name}_lens must have ndim 1 or 2, got {lens.ndim}")
    return indices, lens


def _normalize_sparse_mla_segments(
    *,
    sparse_mla_segments: Sequence[_SparseMLASegment],
    batch_size: int,
    q_len_per_request: int,
    device: torch.device,
) -> List[_NormalizedSparseMLASegment]:
    if not sparse_mla_segments:
        raise ValueError("backend='sparse' requires at least one sparse MLA segment")
    if len(sparse_mla_segments) > 2:
        raise NotImplementedError(
            "backend='sparse' currently supports at most two sparse MLA segments"
        )

    normalized: List[_NormalizedSparseMLASegment] = []
    for i, segment in enumerate(sparse_mla_segments):
        if not isinstance(segment, _SparseMLASegment):
            raise TypeError(
                "sparse_mla_segments must contain _SparseMLASegment instances; "
                f"got {type(segment)!r} at index {i}"
            )
        if i == 0 and segment.kv_cache is not None:
            raise ValueError(
                "sparse_mla_segments[0].kv_cache must be None; the first segment "
                "uses the function's kv_cache argument"
            )
        if i > 0:
            if segment.kv_cache is None:
                raise ValueError(
                    f"sparse_mla_segments[{i}].kv_cache is required for additional "
                    "sparse MLA segments"
                )
            if segment.kv_cache.device != device:
                raise ValueError(
                    f"sparse_mla_segments[{i}].kv_cache must be on device {device}, "
                    f"got {segment.kv_cache.device}"
                )

        indices, lengths = _normalize_sparse_mla_indices_and_lens(
            indices=segment.indices,
            lens=segment.lengths,
            batch_size=batch_size,
            q_len_per_request=q_len_per_request,
            device=device,
            name=f"sparse_mla_segments[{i}].indices",
        )
        normalized.append(
            _NormalizedSparseMLASegment(
                kv_cache=segment.kv_cache,
                indices=indices,
                lengths=lengths,
            )
        )
    return normalized


def _workspace_tensor_view(
    workspace_buffer: torch.Tensor,
    *,
    byte_offset: int,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
) -> Tuple[Optional[torch.Tensor], int]:
    if not workspace_buffer.is_contiguous():
        return None, byte_offset
    elem_size = torch.empty(
        (), dtype=dtype, device=workspace_buffer.device
    ).element_size()
    byte_offset = ((byte_offset + elem_size - 1) // elem_size) * elem_size
    numel = math.prod(shape)
    byte_end = byte_offset + numel * elem_size
    workspace_bytes = workspace_buffer.numel() * workspace_buffer.element_size()
    if byte_end > workspace_bytes:
        return None, byte_offset
    flat = workspace_buffer.view(torch.uint8)
    view = flat[byte_offset:byte_end].view(dtype).view(shape)
    return view, byte_end


def _sparse_mla_decode_workspace(
    workspace_buffer: torch.Tensor,
    *,
    num_tokens: int,
    num_heads: int,
    d_v: int,
    topk: int,
    extra_topk: int,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if num_tokens > 64:
        return None, None
    split_tile = 64
    num_splits = (topk + split_tile - 1) // split_tile + (
        extra_topk + split_tile - 1
    ) // split_tile
    mid_out, offset = _workspace_tensor_view(
        workspace_buffer,
        byte_offset=0,
        shape=(num_tokens, num_heads, num_splits, d_v),
        dtype=torch.bfloat16,
    )
    if mid_out is None:
        return None, None
    mid_lse, _ = _workspace_tensor_view(
        workspace_buffer,
        byte_offset=offset,
        shape=(num_tokens, num_heads, num_splits),
        dtype=torch.float32,
    )
    if mid_lse is None:
        return None, None
    return mid_out, mid_lse


def _trtllm_batch_decode_sparse_mla_sm120(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    workspace_buffer: torch.Tensor,
    sparse_mla_segments: Sequence[_SparseMLASegment],
    out: Optional[torch.Tensor],
    sm_scale: float,
    sinks: Optional[Union[List[torch.Tensor], Tuple[torch.Tensor, ...], torch.Tensor]],
    lse: Optional[torch.Tensor],
    return_lse: bool,
    kv_scale_format: str,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    if not is_sm12x_supported(query.device):
        raise ValueError(
            "SM120 sparse MLA requires SM120a (CUDA >= 12.8) or SM121a (CUDA >= 12.9)"
        )
    if query.ndim != 4:
        raise ValueError(f"Expected query.ndim == 4, got {query.ndim}")
    if kv_cache.device != query.device:
        raise ValueError(
            f"kv_cache must be on query device {query.device}, got {kv_cache.device}"
        )
    if workspace_buffer.device != query.device:
        raise ValueError(
            "workspace_buffer must be on query device "
            f"{query.device}, got {workspace_buffer.device}"
        )

    batch_size, q_len_per_request, num_heads, head_dim = query.shape
    if head_dim not in (512, 576):
        raise ValueError(
            "Sparse MLA supports DSv4 head_dim=512 or "
            f"DSv3.2/GLM head_dim=576, got {head_dim}"
        )
    if num_heads > 128:
        raise ValueError(f"Expected num_heads <= 128, got {num_heads}")

    segments = _normalize_sparse_mla_segments(
        sparse_mla_segments=sparse_mla_segments,
        batch_size=batch_size,
        q_len_per_request=q_len_per_request,
        device=query.device,
    )
    primary_segment = segments[0]
    extra_segment = segments[1] if len(segments) > 1 else None

    from ._sparse_mla_sm120 import _SparseMLAPagedAttentionRunner

    query_flat = query.reshape(batch_size * q_len_per_request, num_heads, head_dim)
    expected_out_shape = (batch_size, q_len_per_request, num_heads, 512)
    if out is None:
        out = torch.empty(expected_out_shape, dtype=torch.bfloat16, device=query.device)
    else:
        check_shape_dtype_device(
            out, expected_out_shape, torch.bfloat16, query.device, "out"
        )
    out_flat = out.view(query_flat.shape[0], num_heads, 512)

    flat_lse_shape = (query_flat.shape[0], num_heads)
    nested_lse_shape = (batch_size, q_len_per_request, num_heads)
    user_lse = lse
    out_lse_arg = None
    if return_lse and lse is not None:
        if tuple(lse.shape) == flat_lse_shape:
            check_shape_dtype_device(
                lse, flat_lse_shape, torch.float32, query.device, "lse"
            )
        elif tuple(lse.shape) == nested_lse_shape:
            check_shape_dtype_device(
                lse, nested_lse_shape, torch.float32, query.device, "lse"
            )
            lse = lse.view(flat_lse_shape)
        else:
            raise ValueError(
                f"lse must have shape {flat_lse_shape} or {nested_lse_shape}; "
                f"got {tuple(lse.shape)}"
            )
        out_lse_arg = lse

    runner = _SparseMLAPagedAttentionRunner(
        max_num_tokens=query_flat.shape[0],
        max_num_heads=num_heads,
        kv_scale_format=kv_scale_format,
        device=query.device,
    )
    extra_topk = extra_segment.indices.shape[-1] if extra_segment is not None else 0
    mid_out, mid_lse = _sparse_mla_decode_workspace(
        workspace_buffer,
        num_tokens=query_flat.shape[0],
        num_heads=num_heads,
        d_v=512,
        topk=primary_segment.indices.shape[-1],
        extra_topk=extra_topk,
    )

    out_lse = runner.run(
        query_flat,
        kv_cache,
        primary_segment.indices,
        out_flat,
        float(sm_scale),
        topk_length=primary_segment.lengths,
        attn_sink=_normalize_optional_mla_sink(sinks, "backend='sparse'"),
        extra_kv_cache=extra_segment.kv_cache if extra_segment is not None else None,
        extra_indices=extra_segment.indices if extra_segment is not None else None,
        extra_topk_length=extra_segment.lengths if extra_segment is not None else None,
        out_lse=out_lse_arg,
        mid_out=mid_out,
        mid_lse=mid_lse,
        return_lse=return_lse,
    )

    if return_lse:
        return out, user_lse if user_lse is not None else out_lse
    return out




def _normalize_dsv4_sparse_mla_kv_cache(
    kv_cache: torch.Tensor,
    kv_layout: Literal["HND", "NHD"],
    name: str,
) -> torch.Tensor:
    if kv_cache.ndim == 3:
        if kv_layout == "NHD":
            raise ValueError(
                f"{name} with ndim == 3 is only valid for kv_layout='HND'; "
                f"got kv_layout={kv_layout}"
            )
        kv_cache = kv_cache.unsqueeze(1)
    elif kv_cache.ndim != 4:
        raise ValueError(f"Expected {name}.ndim == 3 or 4, got {kv_cache.ndim}")

    if kv_layout == "HND":
        # [num_pages, num_kv_heads, page_size, head_dim]
        if kv_cache.size(1) != 1:
            raise ValueError(f"Expected {name}.shape[1] == 1, got {kv_cache.size(1)}")
        return kv_cache
    if kv_layout == "NHD":
        # [num_pages, page_size, num_kv_heads, head_dim] -> HND strides.
        if kv_cache.size(2) != 1:
            raise ValueError(f"Expected {name}.shape[2] == 1, got {kv_cache.size(2)}")
        return kv_cache.transpose(-3, -2)

    raise ValueError(f"kv_layout must be either 'HND' or 'NHD', got {kv_layout}")


def _check_sm120_dsv4_kv_cache_layout(
    kv_cache: torch.Tensor,
    kv_layout: Literal["HND", "NHD"],
    name: str,
) -> torch.Tensor:
    if kv_cache.ndim == 3:
        return kv_cache
    if kv_cache.ndim != 4:
        raise ValueError(f"Expected {name}.ndim == 3 or 4, got {kv_cache.ndim}")
    if kv_layout == "HND":
        if kv_cache.size(1) != 1:
            raise ValueError(
                f"Expected packed SM120 DSV4 HND {name}.shape[1] == 1, got "
                f"{tuple(kv_cache.shape)}"
            )
    elif kv_layout == "NHD":
        if kv_cache.size(2) != 1:
            raise ValueError(
                f"Expected packed SM120 DSV4 NHD {name}.shape[2] == 1, got "
                f"{tuple(kv_cache.shape)}"
            )
    else:
        raise ValueError(f"kv_layout must be either 'HND' or 'NHD', got {kv_layout}")
    return kv_cache


def _normalize_dsv4_topk_lens(
    topk_lens: torch.Tensor,
    batch_size: int,
    q_len_per_request: int,
    sum_seq_q: int,
    name: str,
    device: torch.device,
    cum_seq_lens_q: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if topk_lens.dtype != torch.int32:
        raise ValueError(f"{name} must have dtype torch.int32, got {topk_lens.dtype}")
    if topk_lens.device != device:
        raise ValueError(f"{name} must be on device {device}, got {topk_lens.device}")
    if topk_lens.ndim != 1:
        raise ValueError(f"Expected flattened {name}.ndim == 1, got {topk_lens.ndim}")
    if topk_lens.size(0) != sum_seq_q:
        raise ValueError(
            f"Expected flattened {name}.shape == ({sum_seq_q},), "
            f"got {tuple(topk_lens.shape)}"
        )
    if cum_seq_lens_q is not None:
        check_shape_dtype_device(
            cum_seq_lens_q,
            (batch_size + 1,),
            torch.int32,
            device,
            "cum_seq_lens_q",
        )
    return topk_lens


def _validate_dsv4_sync_checks() -> bool:
    return os.environ.get("FLASHINFER_VALIDATE_INPUTS", "0") not in ("0", "")


def _check_dsv4_sparse_mla_inputs(
    query: torch.Tensor,
    swa_kv_cache: torch.Tensor,
    sparse_indices: torch.Tensor,
    compressed_kv_cache: torch.Tensor,
    sparse_topk_lens: torch.Tensor,
    out: Optional[torch.Tensor],
    sinks: Optional[torch.Tensor],
    kv_layout: Literal["HND", "NHD"],
    cum_seq_lens_q: Optional[torch.Tensor],
    max_q_len: Optional[int],
    *,
    allow_sm120_packed_kv: bool = False,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    int,
    torch.Tensor,
    Tuple[int, ...],
    Optional[torch.Tensor],
]:
    is_varlen_q = cum_seq_lens_q is not None
    out_shape: Tuple[int, ...]
    sparse_indices_prefix_shape: Tuple[int, ...]
    if is_varlen_q:
        if query.ndim != 3:
            raise ValueError(
                "Expected query.ndim == 3 when cum_seq_lens_q is provided, "
                f"got {query.ndim}"
            )
        if cum_seq_lens_q is None:
            raise ValueError("cum_seq_lens_q is required for varlen query input")
        if cum_seq_lens_q.dtype != torch.int32:
            raise ValueError(
                f"cum_seq_lens_q must have dtype torch.int32, got {cum_seq_lens_q.dtype}"
            )
        if cum_seq_lens_q.ndim != 1:
            raise ValueError(
                f"Expected cum_seq_lens_q.ndim == 1, got {cum_seq_lens_q.ndim}"
            )
        batch_size = cum_seq_lens_q.numel() - 1
        if batch_size <= 0:
            raise ValueError(
                f"Expected cum_seq_lens_q.numel() >= 2, got {cum_seq_lens_q.numel()}"
            )
        if cum_seq_lens_q.device != query.device:
            raise ValueError(
                f"cum_seq_lens_q must be on query device {query.device}, "
                f"got {cum_seq_lens_q.device}"
            )
        sum_seq_q, num_heads, head_dim = query.shape
        if max_q_len is None:
            raise ValueError(
                "max_q_len is required when cum_seq_lens_q is provided to avoid "
                "an implicit device-to-host synchronization"
            )
        if max_q_len <= 0:
            raise ValueError(f"Expected max_q_len > 0, got {max_q_len}")
        q_len_per_request = max_q_len
        query_flat = query
        out_shape = (sum_seq_q, num_heads, 512)
        sparse_indices_prefix_shape = (sum_seq_q,)
    else:
        if query.ndim != 4:
            raise ValueError(f"Expected query.ndim == 4, got {query.ndim}")
        batch_size, q_len_per_request, num_heads, head_dim = query.shape
        sum_seq_q = batch_size * q_len_per_request
        if max_q_len is not None and max_q_len != q_len_per_request:
            raise ValueError(
                f"Expected max_q_len == {q_len_per_request} for dense query input, "
                f"got {max_q_len}"
            )
        max_q_len = q_len_per_request
        query_flat = query.flatten(0, 1)
        out_shape = (batch_size, q_len_per_request, num_heads, 512)
        sparse_indices_prefix_shape = (sum_seq_q,)

    if query.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise ValueError(
            "DeepSeek V4 sparse MLA only supports BF16 or FP8 E4M3 query, "
            f"got {query.dtype}"
        )
    if head_dim != 512:
        raise ValueError(f"Expected query head dim 512, got {head_dim}")
    if num_heads not in (8, 16, 32, 64, 128):
        raise ValueError(f"Expected 8, 16, 32, 64, or 128 query heads, got {num_heads}")

    if (
        sparse_indices is None
        or compressed_kv_cache is None
        or sparse_topk_lens is None
    ):
        raise ValueError(
            "sparse_indices, compressed_kv_cache, and sparse_topk_lens are required"
        )
    if sparse_indices.dtype != torch.int32:
        raise ValueError(
            f"sparse_indices must have dtype torch.int32, got {sparse_indices.dtype}"
        )
    if sparse_indices.device != query.device:
        raise ValueError(
            f"sparse_indices must be on query device {query.device}, "
            f"got {sparse_indices.device}"
        )
    if sparse_indices.ndim != 2:
        raise ValueError(
            f"Expected flattened sparse_indices.ndim == 2, got {sparse_indices.ndim}"
        )
    if sparse_indices.shape[:-1] != sparse_indices_prefix_shape:
        raise ValueError(
            "Expected sparse_indices.shape[:-1] == "
            f"{sparse_indices_prefix_shape}, got {sparse_indices.shape[:-1]}"
        )
    if sparse_indices.size(-1) < 128:
        raise ValueError(
            "sparse_indices must include the fixed 128 SWA entries, got "
            f"{sparse_indices.size(-1)}"
        )
    if sparse_indices.size(-1) % 4 != 0:
        raise ValueError(
            "sparse_indices last dimension must be a multiple of 4, got "
            f"{sparse_indices.size(-1)}"
        )

    if allow_sm120_packed_kv:
        swa_kv_cache = _check_sm120_dsv4_kv_cache_layout(
            swa_kv_cache, kv_layout, "swa_kv_cache"
        )
    else:
        swa_kv_cache = _normalize_dsv4_sparse_mla_kv_cache(
            swa_kv_cache, kv_layout, "swa_kv_cache"
        )
    if allow_sm120_packed_kv and swa_kv_cache.dtype == torch.uint8:
        if swa_kv_cache.size(-1) != 584:
            raise ValueError(
                "Expected packed SM120 DSV4 swa_kv_cache head dim 584, got "
                f"{swa_kv_cache.size(-1)}"
            )
    elif swa_kv_cache.dtype != query.dtype:
        raise ValueError(
            f"swa_kv_cache dtype must match query dtype, got {swa_kv_cache.dtype} "
            f"and {query.dtype}"
        )
    elif swa_kv_cache.size(-1) != 512:
        raise ValueError(
            f"Expected swa_kv_cache head dim 512, got {swa_kv_cache.size(-1)}"
        )

    if allow_sm120_packed_kv:
        compressed_kv_cache = _check_sm120_dsv4_kv_cache_layout(
            compressed_kv_cache, kv_layout, "compressed_kv_cache"
        )
    else:
        compressed_kv_cache = _normalize_dsv4_sparse_mla_kv_cache(
            compressed_kv_cache, kv_layout, "compressed_kv_cache"
        )
    if allow_sm120_packed_kv and compressed_kv_cache.dtype == torch.uint8:
        if compressed_kv_cache.size(-1) != 584:
            raise ValueError(
                "Expected packed SM120 DSV4 compressed_kv_cache head dim 584, got "
                f"{compressed_kv_cache.size(-1)}"
            )
    elif compressed_kv_cache.dtype != query.dtype:
        raise ValueError(
            "compressed_kv_cache dtype must match query dtype, got "
            f"{compressed_kv_cache.dtype} and {query.dtype}"
        )
    elif compressed_kv_cache.size(-1) != 512:
        raise ValueError(
            f"Expected compressed_kv_cache head dim 512, got {compressed_kv_cache.size(-1)}"
        )
    normalized_sparse_lens = _normalize_dsv4_topk_lens(
        sparse_topk_lens,
        batch_size,
        q_len_per_request,
        sum_seq_q,
        "sparse_topk_lens",
        query.device,
        cum_seq_lens_q,
    )
    if _validate_dsv4_sync_checks() and normalized_sparse_lens.numel() > 0:
        sparse_topk_capacity = sparse_indices.size(-1)
        invalid_sparse_lens = torch.logical_or(
            normalized_sparse_lens < 128,
            normalized_sparse_lens > sparse_topk_capacity,
        )
        if invalid_sparse_lens.any().item():
            min_sparse_len = int(normalized_sparse_lens.min().item())
            max_sparse_len = int(normalized_sparse_lens.max().item())
            if min_sparse_len < 128:
                raise ValueError(
                    "sparse_topk_lens values must include the fixed 128 SWA entries, "
                    f"got minimum {min_sparse_len}"
                )
            raise ValueError(
                "sparse_topk_lens values cannot exceed sparse_indices capacity, "
                f"got max {max_sparse_len} and capacity {sparse_topk_capacity}"
            )

    if out is not None:
        check_shape_dtype_device(out, out_shape, torch.bfloat16, query.device, "out")
    if sinks is not None:
        check_shape_dtype_device(
            sinks, (num_heads,), torch.float32, query.device, "sinks"
        )

    return (
        swa_kv_cache,
        compressed_kv_cache,
        normalized_sparse_lens,
        batch_size,
        q_len_per_request,
        query_flat,
        out_shape,
        cum_seq_lens_q,
    )


def _resolve_dsv4_sparse_mla_backend(device: torch.device) -> str:
    cc = get_compute_capability(device)
    if cc[0] == 12:
        return "sparse"
    if cc[0] == 10:
        return "trtllm-gen"
    raise ValueError(
        "trtllm_batch_decode_sparse_mla_dsv4 supports SM100/SM103 via "
        f"TRTLLM-GEN or SM120/SM121 via sparse backend, got SM{cc[0]}{cc[1]}"
    )


def _trtllm_batch_decode_sparse_mla_dsv4_sm120(
    *,
    query: torch.Tensor,
    swa_kv_cache: torch.Tensor,
    workspace_buffer: torch.Tensor,
    sparse_indices: torch.Tensor,
    compressed_kv_cache: Optional[torch.Tensor],
    swa_topk_lens: torch.Tensor,
    extra_sparse_indices: Optional[torch.Tensor],
    extra_sparse_topk_lens: Optional[torch.Tensor],
    out: Optional[torch.Tensor],
    bmm1_scale: float,
    bmm2_scale: float,
    sinks: Optional[torch.Tensor],
    kv_layout: Literal["HND", "NHD"],
) -> torch.Tensor:
    if bmm2_scale != 1.0:
        raise ValueError("SM120 DSv4 sparse MLA does not support bmm2_scale")
    if query.ndim in (3, 4):
        num_heads, head_dim = query.shape[-2:]
    else:
        raise ValueError(f"Expected query.ndim == 3 or 4, got {query.ndim}")
    if query.dtype != torch.bfloat16:
        raise ValueError(
            f"SM120 DSv4 sparse MLA only supports BF16 query, got {query.dtype}"
        )
    if head_dim != 512:
        raise ValueError(f"Expected DSv4 query head dim 512, got {head_dim}")
    if num_heads not in (8, 16, 32, 64, 128):
        raise ValueError(
            "Expected 8, 16, 32, 64, or 128 query heads for SM120 DSv4 "
            f"sparse MLA, got {num_heads}"
        )
    if swa_topk_lens is None:
        raise ValueError("backend='sparse' requires swa_topk_lens")

    swa_kv_cache = _check_sm120_dsv4_kv_cache_layout(
        swa_kv_cache, kv_layout, "swa_kv_cache"
    )
    if swa_kv_cache.dtype == torch.uint8:
        if swa_kv_cache.size(-1) != 584:
            raise ValueError(
                "Expected packed SM120 DSV4 swa_kv_cache head dim 584, got "
                f"{swa_kv_cache.size(-1)}"
            )
    elif swa_kv_cache.dtype != query.dtype:
        raise ValueError(
            f"swa_kv_cache dtype must match query dtype, got {swa_kv_cache.dtype} "
            f"and {query.dtype}"
        )
    elif swa_kv_cache.size(-1) != 512:
        raise ValueError(
            f"Expected swa_kv_cache head dim 512, got {swa_kv_cache.size(-1)}"
        )

    if (extra_sparse_indices is None) != (extra_sparse_topk_lens is None):
        raise ValueError(
            "extra_sparse_indices and extra_sparse_topk_lens must be provided "
            "together for backend='sparse'"
        )
    query_for_sm120 = query if query.ndim == 4 else query.unsqueeze(1)
    out_for_sm120 = out if out is None or out.ndim == 4 else out.unsqueeze(1)

    sparse_mla_segments: List[_SparseMLASegment] = [
        _SparseMLASegment(indices=sparse_indices, lengths=swa_topk_lens)
    ]
    if extra_sparse_indices is not None:
        if compressed_kv_cache is None:
            raise ValueError(
                "compressed_kv_cache is required when extra_sparse_indices is provided"
            )
        compressed_kv_cache = _check_sm120_dsv4_kv_cache_layout(
            compressed_kv_cache, kv_layout, "compressed_kv_cache"
        )
        if compressed_kv_cache.dtype == torch.uint8:
            if compressed_kv_cache.size(-1) != 584:
                raise ValueError(
                    "Expected packed SM120 DSV4 compressed_kv_cache head dim 584, "
                    f"got {compressed_kv_cache.size(-1)}"
                )
        elif compressed_kv_cache.dtype != query.dtype:
            raise ValueError(
                "compressed_kv_cache dtype must match query dtype, got "
                f"{compressed_kv_cache.dtype} and {query.dtype}"
            )
        elif compressed_kv_cache.size(-1) != 512:
            raise ValueError(
                "Expected compressed_kv_cache head dim 512, got "
                f"{compressed_kv_cache.size(-1)}"
            )
        sparse_mla_segments.append(
            _SparseMLASegment(
                indices=extra_sparse_indices,
                lengths=extra_sparse_topk_lens,
                kv_cache=compressed_kv_cache,
            )
        )

    result = cast(
        torch.Tensor,
        _trtllm_batch_decode_sparse_mla_sm120(
            query=query_for_sm120,
            kv_cache=swa_kv_cache,
            workspace_buffer=workspace_buffer,
            sparse_mla_segments=sparse_mla_segments,
            out=out_for_sm120,
            sm_scale=bmm1_scale,
            sinks=sinks,
            lse=None,
            return_lse=False,
            kv_scale_format="auto",
        ),
    )
    if query.ndim == 3:
        return result.squeeze(1)
    return result


def trtllm_batch_decode_sparse_mla_dsv4(
    query: torch.Tensor,
    swa_kv_cache: torch.Tensor,
    workspace_buffer: torch.Tensor,
    sparse_indices: torch.Tensor,
    compressed_kv_cache: Optional[torch.Tensor] = None,
    sparse_topk_lens: Optional[torch.Tensor] = None,
    seq_lens: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    bmm1_scale: Union[float, torch.Tensor] = 1.0,
    bmm2_scale: Union[float, torch.Tensor] = 1.0,
    sinks: Optional[torch.Tensor] = None,
    kv_layout: Literal["HND", "NHD"] = "HND",
    cum_seq_lens_q: Optional[torch.Tensor] = None,
    max_q_len: Optional[int] = None,
    enable_pdl: bool | None = None,
    swa_topk_lens: Optional[torch.Tensor] = None,
    extra_sparse_indices: Optional[torch.Tensor] = None,
    extra_sparse_topk_lens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Decode DeepSeek V4 sparse MLA.

    The implementation is selected from the query device architecture.

    On SM100/SM103, this calls the TRTLLM-GEN DeepSeek V4 sparse MLA kernels.
    The query and both KV pools use head dim 512. The query may be BF16 or
    per-tensor FP8 E4M3 and the output is BF16. The first 128 columns of
    ``sparse_indices`` are SWA entries into ``swa_kv_cache``; remaining columns
    are compressed entries into ``compressed_kv_cache``. ``sparse_topk_lens``
    gives the total active sparse length for each query token and must include
    the fixed 128 SWA entries. ``seq_lens`` provides the original KV sequence
    length for the SWA validity window.

    On SM120/SM121, this calls the packed sparse backend. ``swa_kv_cache`` is
    the required packed uint8 SWA pool with 584 bytes per token. ``sparse_indices``
    and ``swa_topk_lens`` describe the active SWA segment. To add a compressed
    segment, pass ``compressed_kv_cache`` as another packed uint8 pool and pass
    ``extra_sparse_indices`` with ``extra_sparse_topk_lens``. The SM120/SM121
    path accepts BF16 query tensors and produces BF16 output.

    Parameters
    ----------
    query : torch.Tensor
        Dense query input ``[batch_size, q_len_per_request, num_heads, 512]``
        or varlen query input ``[sum_q, num_heads, 512]`` when
        ``cum_seq_lens_q`` is provided. SM100/SM103 accepts BF16 or FP8 E4M3;
        SM120/SM121 accepts BF16.
    swa_kv_cache : torch.Tensor
        SWA KV cache. TRTLLM-GEN uses head dim 512; SM120 sparse uses packed
        uint8 head dim 584. Layout follows ``kv_layout``.
    workspace_buffer : torch.Tensor
        TRTLLM-GEN workspace buffer. The multi-CTA KV counters are managed in a
        separate internal buffer.
    sparse_indices : torch.Tensor
        TRTLLM-GEN combined sparse table, or the SM120 sparse SWA segment.
    compressed_kv_cache : Optional[torch.Tensor]
        Primary/compressed KV cache in the same backend layout as
        ``swa_kv_cache``. Required by ``trtllm-gen`` and by SM120 ``sparse``
        when ``extra_sparse_indices`` is provided.
    sparse_topk_lens : Optional[torch.Tensor]
        Flattened total sparse MLA top-k lengths in query-token order, shape
        ``[sum_q]``. Values must already include the fixed 128 SWA entries,
        matching TRTLLM-GEN ``sparseMlaTopkLengths``, and must not exceed
        ``sparse_indices.shape[-1]``. Required only by ``trtllm-gen``.
    seq_lens : Optional[torch.Tensor]
        Original KV sequence lengths, shape ``[batch_size]`` INT32. Required
        only by ``trtllm-gen``.
    bmm1_scale : Union[float, torch.Tensor]
        Fused per-tensor scale for QK and softmax. Tensor form must be FP32.
    bmm2_scale : Union[float, torch.Tensor]
        Fused per-tensor scale for VO. Tensor form must be FP32.
    sinks : Optional[torch.Tensor]
        Optional attention sink logits, shape ``[num_heads]`` FP32.
    kv_layout : Literal["HND", "NHD"]
        Layout of both KV pools.
    cum_seq_lens_q : Optional[torch.Tensor]
        Cumulative query lengths for varlen query input, shape ``[batch_size + 1]``
        INT32. When provided, dynamic top-k lengths are consumed in flattened
        query-token order.
    max_q_len : Optional[int]
        Maximum query length in the varlen batch. Required with
        ``cum_seq_lens_q``.
    enable_pdl : Optional[bool]
        Whether to enable Programmatic Dependent Launch. Used by the
        TRTLLM-GEN path.
    swa_topk_lens : Optional[torch.Tensor]
        Active SWA segment lengths for SM120/SM121, shape ``[sum_q]`` INT32.
    extra_sparse_indices : Optional[torch.Tensor]
        Optional SM120/SM121 compressed segment indices into
        ``compressed_kv_cache``.
    extra_sparse_topk_lens : Optional[torch.Tensor]
        Active compressed segment lengths for SM120/SM121, shape ``[sum_q]``
        INT32.
    """
    backend = _resolve_dsv4_sparse_mla_backend(query.device)
    if enable_pdl is None:
        enable_pdl = device_support_pdl(query.device)
    if isinstance(bmm1_scale, torch.Tensor):
        if backend == "sparse":
            raise ValueError(
                "SM120/SM121 DSv4 sparse MLA expects bmm1_scale to be a float"
            )
        if bmm1_scale.dtype != torch.float32:
            raise TypeError("bmm1_scale tensor must have dtype torch.float32")
        bmm1_scale = bmm1_scale * log2e
    if isinstance(bmm2_scale, torch.Tensor):
        if backend == "sparse":
            raise ValueError(
                "SM120/SM121 DSv4 sparse MLA expects bmm2_scale to be a float"
            )
        if bmm2_scale.dtype != torch.float32:
            raise TypeError("bmm2_scale tensor must have dtype torch.float32")

    if backend == "sparse":
        return _trtllm_batch_decode_sparse_mla_dsv4_sm120(
            query=query,
            swa_kv_cache=swa_kv_cache,
            workspace_buffer=workspace_buffer,
            sparse_indices=sparse_indices,
            compressed_kv_cache=compressed_kv_cache,
            swa_topk_lens=swa_topk_lens,
            extra_sparse_indices=extra_sparse_indices,
            extra_sparse_topk_lens=extra_sparse_topk_lens,
            out=out,
            bmm1_scale=float(bmm1_scale),
            bmm2_scale=float(bmm2_scale),
            sinks=sinks,
            kv_layout=kv_layout,
        )

    if (
        swa_topk_lens is not None
        or extra_sparse_indices is not None
        or extra_sparse_topk_lens is not None
    ):
        raise ValueError(
            "swa_topk_lens, extra_sparse_indices, and extra_sparse_topk_lens "
            "are only supported on SM120/SM121"
        )
    if sparse_topk_lens is None or compressed_kv_cache is None or seq_lens is None:
        raise ValueError(
            "backend='trtllm-gen' requires compressed_kv_cache, sparse_topk_lens, "
            "and seq_lens"
        )

    (
        swa_kv_cache,
        compressed_kv_cache,
        sparse_topk_lens,
        batch_size,
        q_len_per_request,
        query_flat,
        expected_out_shape,
        cum_seq_lens_q,
    ) = _check_dsv4_sparse_mla_inputs(
        query,
        swa_kv_cache,
        sparse_indices,
        compressed_kv_cache,
        sparse_topk_lens,
        out,
        sinks,
        kv_layout,
        cum_seq_lens_q,
        max_q_len,
        allow_sm120_packed_kv=False,
    )

    if out is None:
        out = torch.empty(expected_out_shape, dtype=torch.bfloat16, device=query.device)

    check_shape_dtype_device(
        seq_lens, (batch_size,), torch.int32, query.device, "seq_lens"
    )
    if cum_seq_lens_q is None:
        q_lens = seq_lens.new_full((batch_size,), q_len_per_request)
    else:
        q_lens = cum_seq_lens_q[1:] - cum_seq_lens_q[:-1]
    if _validate_dsv4_sync_checks() and torch.any(seq_lens < q_lens).item():
        raise ValueError(
            "seq_lens must be greater than or equal to the per-request query "
            "lengths so TRTLLM-GEN can derive the SWA-128 valid window"
        )

    primary_kv_cache = compressed_kv_cache
    sparse_indices = sparse_indices.reshape(query_flat.size(0), -1).contiguous()
    sparse_topk_lens = sparse_topk_lens.contiguous()

    op = get_trtllm_gen_fmha_module()
    run_func = getattr(op, "trtllm_paged_attention_decode_sparse_mla_dsv4", None)
    if run_func is None:
        raise RuntimeError(
            "trtllm_paged_attention_decode_sparse_mla_dsv4 is not available. "
            "Rebuild FlashInfer with the DeepSeek V4 sparse MLA TRTLLM-GEN launcher."
        )

    sm_count = get_device_sm_count(query.device)
    # Fresh zero-initialized buffer; the kernel self-resets the counters at the
    # end of the launch, so no explicit re-zeroing is required.
    multi_ctas_kv_counter_buffer = _get_trtllm_gen_multi_ctas_kv_counter_buffer(
        batch_size, query_flat.size(1), sm_count, query.device
    )
    run_func(
        out,
        query_flat,
        primary_kv_cache,
        swa_kv_cache,
        workspace_buffer,
        multi_ctas_kv_counter_buffer,
        sparse_indices,
        seq_lens.contiguous(),
        sparse_topk_lens,
        bmm1_scale,
        bmm2_scale,
        batch_size,
        q_len_per_request,
        sm_count,
        enable_pdl,
        workspace_buffer.numel() * workspace_buffer.element_size(),
        sinks,
        cum_seq_lens_q.contiguous() if cum_seq_lens_q is not None else None,
    )
    return out


_trtllm_batch_decode_sparse_mla_dsv4 = trtllm_batch_decode_sparse_mla_dsv4


# These aliases preserve root `_core` lookups for implementations owned by Batch MLA.
from ._batch_mla._core import (
    BatchMLAPagedAttentionWrapper as _BatchMLAPagedAttentionWrapper,
    CuteDslMlaDecodeRunner as _CuteDslMlaDecodeRunner,
    TrtllmGenMlaDecodeRunner as _TrtllmGenMlaDecodeRunner,
    _run_mla_decode_cute_dsl,
    _run_mla_decode_trtllm_gen,
    _run_mla_decode_trtllm_gen_or_cute_dsl_impl,
    _run_mla_decode_xqa,
    get_batch_mla_module as _get_batch_mla_module,
    get_mla_module as _get_mla_module,
    get_trtllm_gen_fmha_module as _get_trtllm_gen_fmha_module,
    xqa_batch_decode_with_kv_cache_mla as _xqa_batch_decode_with_kv_cache_mla,
)


BatchMLAPagedAttentionWrapper = _BatchMLAPagedAttentionWrapper
CuteDslMlaDecodeRunner = _CuteDslMlaDecodeRunner
TrtllmGenMlaDecodeRunner = _TrtllmGenMlaDecodeRunner
get_batch_mla_module = _get_batch_mla_module
get_mla_module = _get_mla_module
get_trtllm_gen_fmha_module = _get_trtllm_gen_fmha_module
xqa_batch_decode_with_kv_cache_mla = _xqa_batch_decode_with_kv_cache_mla


def _run_mla_decode_sparse(
    *,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    workspace_buffer: torch.Tensor,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    block_tables: torch.Tensor,
    seq_lens: Optional[torch.Tensor],
    max_seq_len: int,
    sparse_mla_top_k: int = 0,
    out: Optional[torch.Tensor] = None,
    bmm1_scale: Union[float, torch.Tensor] = 1.0,
    bmm2_scale: Union[float, torch.Tensor] = 1.0,
    sinks: Optional[List[torch.Tensor]] = None,
    skip_softmax_threshold_scale_factor: Optional[float] = None,
    enable_pdl: bool | None = None,
    backend: str = "auto",
    is_var_seq: bool = True,
    uses_shared_paged_kv_idx: bool = True,
    lse: Optional[torch.Tensor] = None,
    return_lse: bool = False,
    cute_dsl_impl: str = "auto",
    kv_scale_format: str = "auto",
    cum_seq_lens_q: Optional[torch.Tensor] = None,
    max_q_len: Optional[int] = None,
    multi_ctas_kv_counter_buffer: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    if backend != "sparse":
        raise ValueError(
            f"Sparse MLA adapter requires backend='sparse', got {backend!r}"
        )
    if multi_ctas_kv_counter_buffer is not None:
        raise ValueError(
            "multi_ctas_kv_counter_buffer is only supported by the "
            "trtllm-gen backend"
        )
    del (
        max_seq_len,
        cute_dsl_impl,
        enable_pdl,
        is_var_seq,
        cum_seq_lens_q,
        max_q_len,
    )
    return _run_mla_decode_sparse_sm120(
        query=query,
        kv_cache=kv_cache,
        workspace_buffer=workspace_buffer,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        block_tables=block_tables,
        seq_lens=seq_lens,
        sparse_mla_top_k=sparse_mla_top_k,
        out=out,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        sinks=sinks,
        skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
        uses_shared_paged_kv_idx=uses_shared_paged_kv_idx,
        lse=lse,
        return_lse=return_lse,
        kv_scale_format=kv_scale_format,
    )


@flashinfer_api(trace=trtllm_batch_decode_mla_trace_dispatch)
def trtllm_batch_decode_with_kv_cache_mla(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    workspace_buffer: torch.Tensor,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    block_tables: torch.Tensor,
    seq_lens: Optional[torch.Tensor],
    max_seq_len: int,
    sparse_mla_top_k: int = 0,
    out: Optional[torch.Tensor] = None,
    bmm1_scale: Union[float, torch.Tensor] = 1.0,
    bmm2_scale: Union[float, torch.Tensor] = 1.0,
    sinks: Optional[List[torch.Tensor]] = None,
    skip_softmax_threshold_scale_factor: Optional[float] = None,
    enable_pdl: bool | None = None,
    backend: str = "auto",
    is_var_seq: bool = True,
    uses_shared_paged_kv_idx: bool = True,
    lse: Optional[torch.Tensor] = None,
    return_lse: bool = False,
    cute_dsl_impl: str = "auto",
    kv_scale_format: str = "auto",
    cum_seq_lens_q: Optional[torch.Tensor] = None,
    max_q_len: Optional[int] = None,
    multi_ctas_kv_counter_buffer: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    if isinstance(bmm1_scale, torch.Tensor) and bmm1_scale.dtype != torch.float32:
        raise TypeError("bmm1_scale tensor must have dtype torch.float32")
    if isinstance(bmm2_scale, torch.Tensor) and bmm2_scale.dtype != torch.float32:
        raise TypeError("bmm2_scale tensor must have dtype torch.float32")
    if max_q_len is not None and cum_seq_lens_q is None:
        raise ValueError("max_q_len is only supported when cum_seq_lens_q is provided")
    if backend not in ("auto", "xqa", "trtllm-gen", "cute-dsl", "sparse"):
        raise ValueError(f"Backend {backend} not supported")

    if backend == "auto":
        cc = get_compute_capability(query.device)
        if cc[0] == 12 and sparse_mla_top_k > 0:
            backend = "sparse"
        elif cc[0] != 10:
            backend = "xqa"

    adapter = {
        "auto": _run_mla_decode_trtllm_gen,
        "xqa": _run_mla_decode_xqa,
        "trtllm-gen": _run_mla_decode_trtllm_gen,
        "cute-dsl": _run_mla_decode_cute_dsl,
        "sparse": _run_mla_decode_sparse,
    }[backend]
    return adapter(
        query=query,
        kv_cache=kv_cache,
        workspace_buffer=workspace_buffer,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=max_seq_len,
        sparse_mla_top_k=sparse_mla_top_k,
        out=out,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        sinks=sinks,
        skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
        enable_pdl=enable_pdl,
        backend=backend,
        is_var_seq=is_var_seq,
        uses_shared_paged_kv_idx=uses_shared_paged_kv_idx,
        lse=lse,
        return_lse=return_lse,
        cute_dsl_impl=cute_dsl_impl,
        kv_scale_format=kv_scale_format,
        cum_seq_lens_q=cum_seq_lens_q,
        max_q_len=max_q_len,
        multi_ctas_kv_counter_buffer=multi_ctas_kv_counter_buffer,
    )


trtllm_batch_decode_with_kv_cache_mla.__doc__ = (
    _run_mla_decode_trtllm_gen_or_cute_dsl_impl.__doc__
)
