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
import functools
import math
import os
import warnings
from typing import List, Literal, Optional, Sequence, Tuple, Union, cast, overload

import torch

from ..api_logging import flashinfer_api
from ..autotuner import AutoTuner, TunableRunner
from ..trace.templates.attention import (
    mla_paged_decode_trace,
    trtllm_batch_decode_mla_trace_dispatch,
    xqa_batch_decode_mla_trace,
)
from ..jit import gen_batch_mla_module, gen_trtllm_gen_fmha_module, setup_cubin_loader
from ..jit.mla import gen_mla_module
from ..utils import (
    MaskMode,
    _check_block_tables_shape,
    check_shape_dtype_device,
    determine_mla_backend,
    device_support_pdl,
    get_compute_capability,
    get_device_sm_count,
    _get_trtllm_gen_multi_ctas_kv_counter_buffer,
    _resolve_trtllm_gen_multi_ctas_kv_counter_buffer,
    get_trtllm_gen_multi_ctas_kv_counter_bytes,
    is_sm12x_supported,
    log2e,
)
from ..xqa import xqa_mla


def _check_cutlass_shape(q_nope_pe, ckv_kpe_cache, kv_len, page_table):
    if q_nope_pe.ndim != 3:
        raise ValueError(f"Expected q_nope_pe.ndim == 3, got {q_nope_pe.ndim}")
    if ckv_kpe_cache.ndim != 3:
        raise ValueError(f"Expected ckv_kpe_cache.ndim == 3, got {ckv_kpe_cache.ndim}")
    if kv_len.ndim != 1:
        raise ValueError(f"Expected kv_len.ndim == 1, got {kv_len.ndim}")
    if page_table.ndim != 2:
        raise ValueError(f"Expected page_table.ndim == 2, got {page_table.ndim}")
    B_q, H, D_q = q_nope_pe.shape
    D_ckv = ckv_kpe_cache.shape[2]
    if H != 128:
        raise ValueError(f"Expected 128 heads for q_nope_pe, got {H}")
    if D_q != D_ckv or D_q != 576:
        raise ValueError(
            f"Expected head dim 576 for q_nope_pe and ckv_kpe_cache, got {D_q} and {D_ckv}"
        )
    B_block_table, block_num = page_table.shape
    block_size = ckv_kpe_cache.shape[1]
    if B_q != B_block_table:
        raise ValueError(
            f"Expected batch size {B_q} for q_nope_pe and block_table, got {B_q} and {B_block_table}"
        )
    if block_num % (128 / block_size) != 0:
        raise ValueError(
            f"Expected block_num % (128 / block_size) == 0, got {block_num=} and {block_size=}"
        )


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
            "SM120 sparse MLA requires SM120a (CUDA >= 12.8) or SM121a (CUDA >= 13.0)"
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


def _check_sm120_sparse_v32_kv_cache(kv_cache: torch.Tensor) -> torch.Tensor:
    if kv_cache.dtype != torch.uint8:
        raise ValueError(
            "SM120 sparse MLA v32/GLM backend expects packed uint8 kv_cache, "
            f"got {kv_cache.dtype}"
        )
    if kv_cache.ndim == 3:
        if kv_cache.size(-1) != 656:
            raise ValueError(
                "SM120 sparse MLA v32/GLM expects packed kv_cache last dim 656, "
                f"got {tuple(kv_cache.shape)}"
            )
        return kv_cache
    if kv_cache.ndim == 4:
        if kv_cache.size(1) != 1 or kv_cache.size(-1) != 656:
            raise ValueError(
                "SM120 sparse MLA v32/GLM expects public HND kv_cache shape "
                "[num_pages, 1, page_size, 656] or 3D shorthand "
                f"[num_pages, page_size, 656], got {tuple(kv_cache.shape)}"
            )
        return kv_cache
    raise ValueError(f"Expected kv_cache.ndim == 3 or 4, got {kv_cache.ndim}")


def _normalize_sm120_sparse_v32_topk_length(
    seq_lens: Optional[torch.Tensor],
    *,
    batch_size: int,
    q_len_per_request: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if seq_lens is None:
        return None
    if seq_lens.dtype != torch.int32:
        raise ValueError(f"seq_lens must have dtype torch.int32, got {seq_lens.dtype}")
    if seq_lens.device != device:
        raise ValueError(f"seq_lens must be on device {device}, got {seq_lens.device}")
    flat_tokens = batch_size * q_len_per_request
    if seq_lens.ndim == 2 and tuple(seq_lens.shape) == (
        batch_size,
        q_len_per_request,
    ):
        return seq_lens.reshape(flat_tokens).contiguous()
    if seq_lens.ndim == 1 and seq_lens.numel() == flat_tokens:
        return seq_lens.contiguous()
    if seq_lens.ndim == 1 and seq_lens.numel() == batch_size:
        return None
    raise ValueError(
        "seq_lens for SM120 sparse MLA v32/GLM must be shaped either "
        f"({batch_size},), ({flat_tokens},), or "
        f"({batch_size}, {q_len_per_request}); got {tuple(seq_lens.shape)}"
    )


def _trtllm_batch_decode_sparse_mla_v32_sm120(
    *,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    workspace_buffer: torch.Tensor,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    block_tables: torch.Tensor,
    seq_lens: Optional[torch.Tensor],
    sparse_mla_top_k: int,
    out: Optional[torch.Tensor],
    bmm1_scale: Union[float, torch.Tensor],
    bmm2_scale: Union[float, torch.Tensor],
    sinks: Optional[List[torch.Tensor]],
    skip_softmax_threshold_scale_factor: Optional[float],
    uses_shared_paged_kv_idx: bool,
    lse: Optional[torch.Tensor],
    return_lse: bool,
    kv_scale_format: str,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    del qk_nope_head_dim
    if sparse_mla_top_k <= 0:
        raise ValueError("SM120 sparse MLA v32/GLM requires sparse_mla_top_k > 0")
    if skip_softmax_threshold_scale_factor is not None:
        raise ValueError("skip_softmax is not supported for sparse MLA")
    if not uses_shared_paged_kv_idx:
        raise ValueError(
            "SM120 sparse MLA v32/GLM expects shared sparse page indices "
            "(uses_shared_paged_kv_idx=True)"
        )
    if isinstance(bmm1_scale, torch.Tensor):
        raise ValueError("SM120 sparse MLA v32/GLM expects bmm1_scale to be a float")
    if isinstance(bmm2_scale, torch.Tensor):
        raise ValueError("SM120 sparse MLA v32/GLM expects bmm2_scale to be a float")
    if float(bmm2_scale) != 1.0:
        raise ValueError("SM120 sparse MLA v32/GLM does not support bmm2_scale")
    if query.ndim != 4:
        raise ValueError(f"Expected query.ndim == 4, got {query.ndim}")
    if query.dtype != torch.bfloat16:
        raise ValueError(
            f"SM120 sparse MLA v32/GLM expects BF16 query, got {query.dtype}"
        )
    if kv_lora_rank != 512 or qk_rope_head_dim != 64 or query.size(-1) != 576:
        raise ValueError(
            "SM120 sparse MLA v32/GLM expects kv_lora_rank=512, "
            f"qk_rope_head_dim=64, and query head dim 576; got "
            f"kv_lora_rank={kv_lora_rank}, "
            f"qk_rope_head_dim={qk_rope_head_dim}, query dim={query.size(-1)}"
        )
    if workspace_buffer.device != query.device:
        raise ValueError(
            "workspace_buffer must be on query device "
            f"{query.device}, got {workspace_buffer.device}"
        )

    batch_size, q_len_per_request, _, _ = query.shape
    if block_tables.dtype != torch.int32:
        raise ValueError(
            f"block_tables must have dtype torch.int32, got {block_tables.dtype}"
        )
    if block_tables.device != query.device:
        raise ValueError(
            f"block_tables must be on query device {query.device}, "
            f"got {block_tables.device}"
        )
    expected_block_tables_shape = (batch_size, q_len_per_request, sparse_mla_top_k)
    if tuple(block_tables.shape) != expected_block_tables_shape:
        raise ValueError(
            "SM120 sparse MLA v32/GLM expects sparse block_tables shape "
            f"{expected_block_tables_shape}, got {tuple(block_tables.shape)}"
        )

    kv_cache = _check_sm120_sparse_v32_kv_cache(kv_cache)
    topk_length = _normalize_sm120_sparse_v32_topk_length(
        seq_lens,
        batch_size=batch_size,
        q_len_per_request=q_len_per_request,
        device=query.device,
    )
    sparse_segment = _SparseMLASegment(
        indices=block_tables,
        lengths=topk_length,
    )
    return _trtllm_batch_decode_sparse_mla_sm120(
        query=query,
        kv_cache=kv_cache,
        workspace_buffer=workspace_buffer,
        sparse_mla_segments=[sparse_segment],
        out=out,
        sm_scale=float(bmm1_scale),
        sinks=sinks,
        lse=lse,
        return_lse=return_lse,
        kv_scale_format=kv_scale_format,
    )


def _check_trtllm_gen_mla_shape(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    sparse_mla_top_k: int,
    page_table: torch.Tensor,
    page_size: int,
    uses_shared_paged_kv_idx: bool = True,
    batch_size: Optional[int] = None,
    max_q_len: Optional[int] = None,
    require_aligned_block_table: bool = True,
) -> torch.Tensor:
    is_flattened_query = False
    if query.ndim == 4:
        num_seqs, num_tokens, _, qk_head_dim = query.shape
    elif query.ndim == 3:
        is_flattened_query = True
        if batch_size is None or max_q_len is None:
            raise ValueError(
                "batch_size and max_q_len are required when query.ndim == 3"
            )
        num_seqs = batch_size
        num_tokens = max_q_len
        _, _, qk_head_dim = query.shape
    else:
        raise ValueError(f"Expected query.ndim == 3 or 4, got {query.ndim}")

    # Support both 3D and 4D kv_cache for backward compatibility
    if kv_cache.ndim == 3:
        # [num_pages, page_size, head_dim_ckv + head_dim_kpe] -> [num_pages, 1, page_size, head_dim_ckv + head_dim_kpe]
        kv_cache = kv_cache.unsqueeze(1)
    elif kv_cache.ndim != 4:
        raise ValueError(f"Expected kv_cache.ndim == 3 or 4, got {kv_cache.ndim}")

    is_deepseek_dimensions = (
        kv_lora_rank == deepseek_mla_dimensions.kv_lora_rank
        and qk_rope_head_dim == deepseek_mla_dimensions.qk_rope_head_dim
    )
    is_smaller_mla_dimensions = (
        kv_lora_rank == smaller_mla_dimensions.kv_lora_rank
        and qk_rope_head_dim == smaller_mla_dimensions.qk_rope_head_dim
    )
    if not (is_deepseek_dimensions or is_smaller_mla_dimensions):
        raise ValueError(
            f"Unsupported MLA dimensions, got kv_lora_rank={kv_lora_rank} and qk_rope_head_dim={qk_rope_head_dim}, supported dimensions are: {supported_mla_head_dimensions}"
        )

    ckv_dim = kv_cache.shape[3]
    expected_qk_head_dim = kv_lora_rank + qk_rope_head_dim
    if qk_head_dim != expected_qk_head_dim or ckv_dim != expected_qk_head_dim:
        raise ValueError(
            f"Expected head dim {expected_qk_head_dim} for query and kv_cache, got {qk_head_dim} and {ckv_dim}"
        )

    if sparse_mla_top_k > 0:
        page_table_shape = page_table.shape
        expected_page_table_shape = (
            (query.size(0), sparse_mla_top_k)
            if is_flattened_query
            else (num_seqs, num_tokens, sparse_mla_top_k)
        )
        if page_table_shape != expected_page_table_shape:
            raise ValueError(
                "Expected page_table.shape == "
                f"{expected_page_table_shape}" + f", got {page_table_shape}"
            )
    else:
        _check_block_tables_shape(page_table, uses_shared_paged_kv_idx)
        B_block_table = page_table.shape[0]
        block_num = page_table.shape[-1]
        block_size = page_size
        if num_seqs != B_block_table:
            raise ValueError(
                f"Expected batch size {num_seqs} for query and block_table, got {num_seqs} and {B_block_table}"
            )
        if require_aligned_block_table and block_num % (128 / block_size) != 0:
            raise ValueError(
                f"Expected block_num % (128 / block_size) == 0, got {block_num=} and {block_size=}"
            )

    return kv_cache


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


@functools.cache
def get_trtllm_gen_fmha_module():
    mod = gen_trtllm_gen_fmha_module()
    op = mod.build_and_load()
    setup_cubin_loader(mod.get_library_path())
    return op


@functools.cache
def get_mla_module():
    return gen_mla_module().build_and_load()


@functools.cache
def get_batch_mla_module(backend, *args):
    return gen_batch_mla_module(backend, *args).build_and_load()


class BatchMLAPagedAttentionWrapper:
    r"""Wrapper class for MLA (`Multi-head Latent Attention <https://arxiv.org/abs/2405.04434>`_)
    PagedAttention on DeepSeek models. This kernel can be used in decode, and incremental prefill
    and should be used together with `Matrix Absorption trick
    <https://github.com/madsys-dev/deepseekv2-profile/blob/main/workspace/blog/optimizing-mla.md>`_:
    where :math:`W_{UQ}` is absorbed with :math:`W_{UK}`, and :math:`W_{UV}` is
    absorbed with :math:`W_{O}`.
    For MLA attention without Matrix Absorption (``head_dim_qk=192`` and ``head_dim_vo=128``, which is
    used in prefilling self-attention stage), please use
    :class:`flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper`.

    More information about The Paged KV-Cache layout in MLA is explained in our tutorial
    :ref:`MLA Page Layout <mla-page-layout>`.

    For more details about the MLA computation, Matrix Absorption and FlashInfer's MLA implementation,
    please refer to our `blog post <http://flashinfer.ai/2025/02/10/flashinfer-deepseek-mla.html>`_.

    Example
    -------
    >>> import torch
    >>> import flashinfer
    >>> num_local_heads = 128
    >>> batch_size = 114
    >>> head_dim_ckv = 512
    >>> head_dim_kpe = 64
    >>> page_size = 1
    >>> mla_wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
    ...     torch.empty(128 * 1024 * 1024, dtype=torch.int8).to(0),
    ...     backend="fa2"
    ... )
    >>> q_indptr = torch.arange(0, batch_size + 1).to(0).int() # for decode, each query length is 1
    >>> kv_lens = torch.full((batch_size,), 999, dtype=torch.int32).to(0)
    >>> kv_indptr = torch.arange(0, batch_size + 1).to(0).int() * 999
    >>> kv_indices = torch.arange(0, batch_size * 999).to(0).int()
    >>> q_nope = torch.randn(
    ...     batch_size * 1, num_local_heads, head_dim_ckv, dtype=torch.bfloat16, device="cuda"
    ... )
    >>> q_pe = torch.zeros(
    ...     batch_size * 1, num_local_heads, head_dim_kpe, dtype=torch.bfloat16, device="cuda"
    ... )
    >>> ckv = torch.randn(
    ...     batch_size * 999, 1, head_dim_ckv, dtype=torch.bfloat16, device="cuda"
    ... )
    >>> kpe = torch.zeros(
    ...     batch_size * 999, 1, head_dim_kpe, dtype=torch.bfloat16, device="cuda"
    ... )
    >>> sm_scale = 1.0 / ((128 + 64) ** 0.5)  # use head dimension before matrix absorption
    >>> mla_wrapper.plan(
    ...     q_indptr,
    ...     kv_indptr,
    ...     kv_indices,
    ...     kv_lens,
    ...     num_local_heads,
    ...     head_dim_ckv,
    ...     head_dim_kpe,
    ...     page_size,
    ...     False,  # causal
    ...     sm_scale,
    ...     q_nope.dtype,
    ...     ckv.dtype,
    ... )
    >>> o = mla_wrapper.run(q_nope, q_pe, ckv, kpe, return_lse=False)
    >>> o.shape
    torch.Size([114, 128, 512])
    """

    _blackwell_auto_fallback_warned: bool = False

    @classmethod
    def _maybe_warn_blackwell_auto_fallback(
        cls, device: torch.device, selected_backend: str
    ) -> None:
        if cls._blackwell_auto_fallback_warned:
            return
        major, minor = get_compute_capability(device)
        if major < 10:
            return
        cls._blackwell_auto_fallback_warned = True
        warnings.warn(
            f"BatchMLAPagedAttentionWrapper: backend='auto' selected "
            f"'{selected_backend}' on SM{major}{minor}, which is not Blackwell-native "
            f"and gives poor MLA decode performance. "
            f"For decode, use "
            f"flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla "
            f"(Blackwell-native trtllm-gen); backend='cutlass' is the closest "
            f"in-wrapper alternative but may be slower than this fallback for "
            f"decode shapes.",
            UserWarning,
            stacklevel=3,
        )

    @flashinfer_api
    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        use_cuda_graph: bool = False,
        qo_indptr: Optional[torch.Tensor] = None,
        kv_indptr: Optional[torch.Tensor] = None,
        kv_indices: Optional[torch.Tensor] = None,
        kv_len_arr: Optional[torch.Tensor] = None,
        backend: str = "auto",
    ) -> None:
        r"""Constructor for BatchMLAPagedAttentionWrapper.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor
            The user reserved workspace buffer used to store intermediate attention results in
            split-k algorithm. The recommended size is 128MB, the device of the workspace buffer
            should be the same as the device of the input tensors.
        use_cuda_graph : bool, optional
            Whether to enable CUDA graph capture for the prefill kernels, if enabled, the
            auxiliary data structures will be stored in provided buffers. The ``batch_size``
            cannot change during the lifecycle of this wrapper when CUDAGraph is enabled.
        qo_indptr : Optional[torch.Tensor]
            User-reserved buffer to back the ``qo_indptr`` array, shape ``[batch_size + 1]``,
            dtype ``int32``.  Only consulted when ``use_cuda_graph=True``.  The wrapper
            copies into this buffer at :meth:`plan` time so capture-time pointers remain
            stable.
        kv_indptr : Optional[torch.Tensor]
            User-reserved buffer to back the ``kv_indptr`` array, shape ``[batch_size + 1]``,
            dtype ``int32``.  Only consulted when ``use_cuda_graph=True``.
        kv_indices : Optional[torch.Tensor]
            User-reserved buffer to back the ``kv_indices`` array, sized to the maximum
            expected number of pages, dtype ``int32``.  Only consulted when
            ``use_cuda_graph=True``.
        kv_len_arr : Optional[torch.Tensor]
            User-reserved buffer to back the ``kv_len_arr`` array, shape ``[batch_size]``,
            dtype ``int32``.  Only consulted when ``use_cuda_graph=True``.
        backend : str
            One of ``"auto"``, ``"fa2"``, ``"fa3"``, ``"cutlass"``. Default ``"auto"``.

            ``"auto"`` picks ``"fa3"`` on SM90a, else ``"fa2"``. On SM>=100 neither
            is Blackwell-native; for MLA decode prefer
            :func:`trtllm_batch_decode_with_kv_cache_mla`. The ``"cutlass"`` option
            in this wrapper is the closest in-wrapper alternative but may be
            slower than the fa2 fallback for decode shapes.

            ``"cutlass"`` uses the SM100/SM110 CUTLASS MLA decode kernel. Only
            ``float_workspace_buffer`` is required; ``run()`` takes a different
            input layout (concatenated ``q_nope_pe`` / ``ckv_kpe_cache`` plus
            ``kv_len`` and ``page_table``).
        """
        self._float_workspace_buffer = float_workspace_buffer
        self.device = float_workspace_buffer.device

        if backend == "cutlass":
            self._backend = backend
            return

        self._int_workspace_buffer = torch.empty(
            (8 * 1024 * 1024,), dtype=torch.uint8, device=self.device
        )
        self._pin_memory_int_workspace_buffer = torch.empty(
            self._int_workspace_buffer.shape,
            dtype=self._int_workspace_buffer.dtype,
            pin_memory=True,
            device="cpu",
        )
        self._use_cuda_graph = use_cuda_graph
        self._qo_indptr_buf = qo_indptr
        self._kv_indptr_buf = kv_indptr
        self._kv_indices_buf = kv_indices
        self._kv_len_arr_buf = kv_len_arr
        if backend == "auto":
            self._backend = determine_mla_backend(self.device)
            self._maybe_warn_blackwell_auto_fallback(self.device, self._backend)
        else:
            self._backend = backend

    @flashinfer_api
    def plan(
        self,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        kv_len_arr: torch.Tensor,
        num_heads: int,
        head_dim_ckv: int,
        head_dim_kpe: int,
        page_size: int,
        causal: bool,
        sm_scale: float,
        q_data_type: torch.dtype,
        kv_data_type: torch.dtype,
        use_profiler: bool = False,
    ) -> None:
        r"""Plan the MLA attention computation.

        Parameters
        ----------
        qo_indptr : torch.IntTensor
            The indptr of the query/output tensor, shape: ``[batch_size + 1]``.
            For decoding attention, the length of each query is 1, and the content
            of the tensor should be ``[0, 1, 2, ..., batch_size]``.
        kv_indptr : torch.IntTensor
            The indptr of the paged kv-cache, shape: ``[batch_size + 1]``.
        kv_indices : torch.IntTensor
            The page indices of the paged kv-cache, shape: ``[kv_indptr[-1]]`` or larger.
        kv_len_arr : torch.IntTensor
            The query length of each request, shape: ``[batch_size]``.
        num_heads : int
            The number of heads in query/output tensor.
        head_dim_ckv : int
            The head dimension of compressed-kv.
        head_dim_kpe : int
            The head dimension for rope k-cache.
        page_size : int
            The page size of the paged kv-cache.
        causal : bool
            Whether to use causal attention.
        sm_scale : float
            The scale factor for softmax operation.
        q_data_type : torch.dtype
            The data type of the query tensor.
        kv_data_type : torch.dtype
            The data type of the kv-cache tensor.
        use_profiler : bool, optional
            Whether to enable intra-kernel profiler, default is False.
        """
        # Other 1-byte dtypes (uint8, fp4, e5m2) would JIT-map to non-FP8
        # element types and silently take an unsupported code path inside
        # the kernel, so allowlist exactly the dtypes the kernel can handle.
        _SUPPORTED_MLA_KV_DTYPES = (torch.float16, torch.bfloat16, torch.float8_e4m3fn)
        if kv_data_type not in _SUPPORTED_MLA_KV_DTYPES:
            raise ValueError(
                f"MLA kv_data_type {kv_data_type} is not supported. "
                f"Supported dtypes: {list(_SUPPORTED_MLA_KV_DTYPES)}."
            )
        if kv_data_type == torch.float8_e4m3fn:
            if self._backend != "fa3":
                raise ValueError(
                    "FP8 kv_data_type for MLA is only supported with the fa3 "
                    f"backend on SM90, got backend={self._backend!r}."
                )
            # Backend selection is independent of the runtime device; FP8 MLA
            # requires SM90 specifically.
            major, minor = get_compute_capability(self.device)
            if major != 9:
                raise ValueError(
                    "FP8 kv_data_type for MLA requires an SM90 (Hopper) device, "
                    f"got SM{major}{minor}."
                )
            # Removing this guard exposes vec_cast<half, fp8_e4m3>, which
            # exists but is untested for MLA — silent wrong output.
            if q_data_type != torch.bfloat16:
                raise ValueError(
                    "FP8 kv_data_type for MLA currently only supports "
                    f"q_data_type=torch.bfloat16, got {q_data_type}."
                )
            # Also enforced by static_assert in mla_hopper.cuh.
            if head_dim_ckv != 512 or head_dim_kpe != 64:
                raise ValueError(
                    "FP8 kv_data_type for MLA currently only supports "
                    "head_dim_ckv=512 and head_dim_kpe=64 (DeepSeek MLA), got "
                    f"head_dim_ckv={head_dim_ckv}, head_dim_kpe={head_dim_kpe}."
                )

        self._cached_module = get_batch_mla_module(
            self._backend,
            q_data_type,
            kv_data_type,
            q_data_type,
            qo_indptr.dtype,
            head_dim_ckv,
            head_dim_kpe,
            use_profiler,
        )
        qo_indptr_host = qo_indptr.to("cpu")
        kv_indptr_host = kv_indptr.to("cpu")
        kv_len_arr_host = kv_len_arr.to("cpu")

        if self._use_cuda_graph:
            self._qo_indptr_buf.copy_(qo_indptr, non_blocking=True)
            self._kv_indptr_buf.copy_(kv_indptr, non_blocking=True)
            self._kv_indices_buf[: len(kv_indices)].copy_(kv_indices, non_blocking=True)
            self._kv_len_arr_buf.copy_(kv_len_arr, non_blocking=True)
        else:
            self._qo_indptr_buf = qo_indptr.to(self.device, non_blocking=True)
            self._kv_indptr_buf = kv_indptr.to(self.device, non_blocking=True)
            self._kv_indices_buf = kv_indices.to(self.device, non_blocking=True)
            self._kv_len_arr_buf = kv_len_arr.to(self.device, non_blocking=True)
        self._causal = causal
        self._page_size = page_size
        self._sm_scale = sm_scale
        # Used by run() to reject dtype mismatches; the C++ launcher
        # reinterprets storage by the JIT-template type chosen at plan(),
        # so a mismatch produces silent wrong output.
        self._q_data_type = q_data_type
        self._kv_data_type = kv_data_type
        self._use_profiler = use_profiler

        self._plan_info = self._cached_module.plan(
            self._float_workspace_buffer,
            self._int_workspace_buffer,
            self._pin_memory_int_workspace_buffer,
            qo_indptr_host,
            kv_indptr_host,
            kv_len_arr_host,
            num_heads,
            head_dim_ckv,  # head_dim_o
            causal,
        )

    @overload
    def run(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        ckv_cache: torch.Tensor,
        kpe_cache: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: Literal[False] = False,
        profiler_buffer: Optional[torch.Tensor] = None,
        kv_len: Optional[torch.Tensor] = None,
        page_table: Optional[torch.Tensor] = None,
        return_lse_base_on_e: bool = False,
        o_scale: Optional[float] = None,
        *,
        ckv_scale: Optional[float] = None,
        kpe_scale: Optional[float] = None,
    ) -> torch.Tensor: ...

    @overload
    def run(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        ckv_cache: torch.Tensor,
        kpe_cache: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: Literal[True] = True,
        profiler_buffer: Optional[torch.Tensor] = None,
        kv_len: Optional[torch.Tensor] = None,
        page_table: Optional[torch.Tensor] = None,
        return_lse_base_on_e: bool = False,
        o_scale: Optional[float] = None,
        *,
        ckv_scale: Optional[float] = None,
        kpe_scale: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...

    @flashinfer_api(trace=mla_paged_decode_trace)
    def run(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        ckv_cache: torch.Tensor,
        kpe_cache: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: bool = False,
        profiler_buffer: Optional[torch.Tensor] = None,
        kv_len: Optional[torch.Tensor] = None,
        page_table: Optional[torch.Tensor] = None,
        return_lse_base_on_e: bool = False,
        o_scale: Optional[float] = None,
        *,
        ckv_scale: Optional[float] = None,
        kpe_scale: Optional[float] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""Run the MLA attention computation.

        Parameters
        ----------
        q_nope : torch.Tensor
            The query tensor without rope, shape: ``[batch_size, num_heads, head_dim_ckv]``.
        q_pe : torch.Tensor
            The rope part of the query tensor, shape: ``[batch_size, num_heads, head_dim_kpe]``.
        ckv_cache : torch.Tensor
            The compressed kv-cache tensor (without rope), shape: ``[num_pages, page_size, head_dim_ckv]``.
            ``head_dim_ckv`` is 512 in DeepSeek v2/v3 models.
        kpe_cache : torch.Tensor
            The rope part of the kv-cache tensor, shape: ``[num_pages, page_size, head_dim_kpe]``.
            ``head_dim_kpe`` is 64 in DeepSeek v2/v3 models.
        out : Optional[torch.Tensor]
            The output tensor, if not provided, will be allocated internally.
            When ``o_scale`` is provided, this should be an FP8 tensor.
        lse : Optional[torch.Tensor]
            The log-sum-exp of attention logits, if not provided, will be allocated internally.
        return_lse : bool, optional
            Whether to return the log-sum-exp value, default is False.
        profiler_buffer : Optional[torch.Tensor]
            The buffer to store the profiler data.
        kv_len : Optional[torch.Tensor]
            The query length of each request, shape: ``[batch_size]``. Required when ``backend`` is ``cutlass``.
        page_table : Optional[torch.Tensor]
            The page table of the paged kv-cache, shape: ``[batch_size, num_pages]``. Required when ``backend`` is ``cutlass``.
        return_lse_base_on_e : bool, optional
            Controls the base of the returned LSE values when ``return_lse=True``.
            If ``False`` (default), the LSE is returned in base-2
            (``log2(sum(exp2(...)))``) to match the kernel's internal log-base.
            If ``True``, the LSE is converted to natural-log base (``log(sum(exp(...)))``)
            for compatibility with cascade-merging APIs that expect base-e LSEs.
        o_scale : Optional[float]
            FP8 output dequantization scale (``real = quantized * o_scale``).
            When provided, ``out`` must be an FP8 tensor. Only supported with
            the ``cutlass`` backend.
        ckv_scale : Optional[float]
            Per-tensor dequantization scale for the compressed-KV cache when
            ``kv_data_type`` is FP8 (``real = quantized * ckv_scale``). Required
            (together with ``kpe_scale``) for the FP8 KV cache path on the
            ``fa3`` backend. Must be a finite positive value. Must not be
            provided when ``kv_data_type`` is BF16/FP16.
        kpe_scale : Optional[float]
            Per-tensor dequantization scale for the rope-K cache when
            ``kv_data_type`` is FP8 (``real = quantized * kpe_scale``). Same
            usage rules as ``ckv_scale``.
        """
        if self._backend == "cutlass":
            if return_lse:
                raise ValueError("return_lse does not support cutlass backend for now.")
            if profiler_buffer is not None:
                raise ValueError(
                    "profiler_buffer does not support cutlass backend for now."
                )
            if ckv_scale is not None or kpe_scale is not None:
                raise ValueError(
                    "ckv_scale / kpe_scale are only supported with the fa3 backend "
                    "and FP8 kv_data_type."
                )
            self._cached_module = get_mla_module()
            output_scale = 1.0
            if o_scale is not None:
                output_scale = float(o_scale)
                if not math.isfinite(output_scale) or output_scale <= 0.0:
                    raise ValueError(
                        f"o_scale must be a finite positive value, got {o_scale}"
                    )
                if out is None:
                    raise ValueError(
                        "out tensor must be provided when o_scale is used for FP8 output."
                    )
                if out.dtype not in (
                    torch.float8_e4m3fn,
                    torch.float8_e5m2,
                ):
                    raise ValueError(
                        f"out must be an FP8 tensor when o_scale is provided, got {out.dtype}"
                    )
                check_shape_dtype_device(out, q_nope.shape, None, q_nope.device, "out")
            elif out is None:
                out = torch.empty_like(q_nope)
            else:
                check_shape_dtype_device(
                    out, q_nope.shape, q_nope.dtype, q_nope.device, "out"
                )
            q_nope_pe = torch.cat([q_nope, q_pe], dim=-1)
            ckv_kpe_cache = torch.cat([ckv_cache, kpe_cache], dim=-1)
            _check_cutlass_shape(q_nope_pe, ckv_kpe_cache, kv_len, page_table)
            lse = torch.empty(0, dtype=torch.float32, device=self.device)
            self._cached_module.cutlass_mla_paged_attention(
                self._float_workspace_buffer,
                out,
                lse,
                q_nope_pe,
                ckv_kpe_cache,
                kv_len,
                page_table,
                output_scale,
            )
            return out

        if o_scale is not None:
            raise ValueError(
                "o_scale is only supported with the cutlass backend for now."
            )

        # The C++ launcher reinterprets tensor storage by the JIT-template
        # type chosen at plan(); a dtype mismatch here silently produces
        # wrong output.
        if q_nope.dtype != self._q_data_type:
            raise ValueError(
                f"q_nope.dtype={q_nope.dtype} does not match the planned "
                f"q_data_type={self._q_data_type}."
            )
        if q_pe.dtype != self._q_data_type:
            raise ValueError(
                f"q_pe.dtype={q_pe.dtype} does not match the planned "
                f"q_data_type={self._q_data_type}."
            )
        if ckv_cache.dtype != self._kv_data_type:
            raise ValueError(
                f"ckv_cache.dtype={ckv_cache.dtype} does not match the planned "
                f"kv_data_type={self._kv_data_type}."
            )
        if kpe_cache.dtype != self._kv_data_type:
            raise ValueError(
                f"kpe_cache.dtype={kpe_cache.dtype} does not match the planned "
                f"kv_data_type={self._kv_data_type}."
            )

        # e4m3fn is the only FP8 dtype reachable here (plan() rejects others).
        kv_is_fp8 = self._kv_data_type == torch.float8_e4m3fn
        if kv_is_fp8:
            if ckv_scale is None or kpe_scale is None:
                raise ValueError(
                    "ckv_scale and kpe_scale are required when kv_data_type is FP8."
                )
            ckv_scale_f = float(ckv_scale)
            kpe_scale_f = float(kpe_scale)
            if not math.isfinite(ckv_scale_f) or ckv_scale_f <= 0.0:
                raise ValueError(
                    f"ckv_scale must be a finite positive value, got {ckv_scale}"
                )
            if not math.isfinite(kpe_scale_f) or kpe_scale_f <= 0.0:
                raise ValueError(
                    f"kpe_scale must be a finite positive value, got {kpe_scale}"
                )
        else:
            if ckv_scale is not None or kpe_scale is not None:
                raise ValueError(
                    "ckv_scale / kpe_scale are only valid when kv_data_type is FP8."
                )
            ckv_scale_f = 1.0
            kpe_scale_f = 1.0

        if profiler_buffer is None:
            if self._use_profiler:
                raise ValueError(
                    "Profiler is enabled, profiler_buffer must be provided"
                )
        num_heads = q_nope.shape[1]
        page_size = self._page_size
        sm_scale = self._sm_scale
        causal = self._causal
        mask_mode = MaskMode.CAUSAL.value if causal else MaskMode.NON_CAUSAL.value
        device = self.device
        if out is None:
            out = torch.empty_like(q_nope)
        else:
            check_shape_dtype_device(
                out, q_nope.shape, q_nope.dtype, q_nope.device, "out"
            )

        if return_lse:
            if lse is None:
                lse = torch.empty(q_nope.shape[:2], dtype=torch.float32, device=device)
            else:
                check_shape_dtype_device(
                    lse, q_nope.shape[:2], torch.float32, q_nope.device, "lse"
                )
        profiler_args = (profiler_buffer,) if self._use_profiler else ()
        self._cached_module.run(
            self._float_workspace_buffer,
            self._int_workspace_buffer,
            self._plan_info,
            q_nope,
            q_pe,
            ckv_cache,
            kpe_cache,
            self._kv_indices_buf,
            out,
            lse,
            mask_mode,
            num_heads,
            page_size,
            sm_scale,
            return_lse_base_on_e,
            ckv_scale_f,
            kpe_scale_f,
            *profiler_args,
        )

        return (out, lse) if return_lse else out


# ---------------------------------------------------------------------------
# Autotuning support for trtllm_batch_decode_with_kv_cache_mla
# ---------------------------------------------------------------------------

# Keep the trtllm-gen autotune sweep bounded; actual counter storage is sized
# dynamically per profiled batch.
_TRTLLM_GEN_MLA_MAX_BATCH = 8192


def _round_to_seq_len_bucket(x: int) -> int:
    """Power-of-2 bucket for max_seq_len in autotune cache keys.

    Collapses small variations across deployments so they can share cache
    entries (e.g., max_seq_len=16384 and 16385 hash to the same bucket).
    """
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def _cute_dsl_max_supported_batch(
    workspace_bytes: int,
    q_len: int,
    num_heads: int,
    kv_lora_rank: int,
    max_active_blocks: int,
    candidate_max: int,
) -> int:
    """Largest batch the caller's workspace can support for cute-dsl MLA decode.

    Cute-dsl's workspace requirement grows with B via the kernel's per-CTA
    split-K state. Binary-search for the largest ``B <= candidate_max`` whose
    ``get_workspace_size(...)`` fits in ``workspace_bytes``.
    """
    from ..cute_dsl.attention.wrappers.batch_mla import (
        _get_split_kv_and_workspace_size,
    )

    lo, hi = 1, max(1, candidate_max)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        _, ws = _get_split_kv_and_workspace_size(
            mid, q_len, num_heads, kv_lora_rank, max_active_blocks
        )
        if ws <= workspace_bytes:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _compute_mla_decode_buckets(
    workspace_buffer: torch.Tensor,
    runner_names: List[str],
    q_len: int,
    num_heads: int,
    kv_lora_rank: int,
    device: torch.device,
) -> Tuple[int, ...]:
    """Compute the autotune bucket list from kernel/workspace limits only.

    The cap is intentionally independent of the caller's ``query.shape[0]``
    and ``kv_cache.shape[0]`` so that autotuning at any single batch size
    still tunes the full range a runner can actually serve. Buckets above a
    runner's individual cap are handled by that runner's ``get_valid_tactics``
    returning ``[]`` for the affected profiles.

    Caps:
      - trtllm-gen kernel-hardcoded ceiling: 8192
      - cute-dsl dynamic workspace cap (via binary search)
    """
    from ..fused_moe.utils import get_hybrid_num_tokens_buckets

    cap = 0
    if "trtllm-gen" in runner_names:
        cap = max(cap, _TRTLLM_GEN_MLA_MAX_BATCH)
    if "cute-dsl" in runner_names:
        from ..cute_dsl.utils import get_num_sm

        cute_dsl_cap = _cute_dsl_max_supported_batch(
            workspace_bytes=workspace_buffer.numel() * workspace_buffer.element_size(),
            q_len=q_len,
            num_heads=num_heads,
            kv_lora_rank=kv_lora_rank,
            max_active_blocks=get_num_sm(device),
            candidate_max=_TRTLLM_GEN_MLA_MAX_BATCH,
        )
        cap = max(cap, cute_dsl_cap)

    return get_hybrid_num_tokens_buckets(max(1, cap))


def _cute_dsl_incompatibility_reason(
    query: torch.Tensor,
    out_dtype: torch.dtype,
    bmm1_scale: Union[float, torch.Tensor],
    bmm2_scale: Union[float, torch.Tensor],
    sinks: Optional[List[torch.Tensor]],
    sparse_mla_top_k: int,
    skip_softmax_threshold_scale_factor: Optional[float],
    uses_shared_paged_kv_idx: bool,
    qk_rope_head_dim: int,
    kv_lora_rank: int,
    page_size: int,
    is_var_seq: bool,
    return_lse: bool,
    lse: Optional[torch.Tensor],
    cute_dsl_impl: str = "auto",
) -> Optional[str]:
    """Return None if cute-dsl can handle this call, else a human-readable reason.

    Used by both the explicit ``backend="cute-dsl"`` path (raises the reason
    as a ``ValueError``) and the ``backend="auto"`` filter (silently drops
    cute-dsl from the runners list).
    """
    cc = get_compute_capability(query.device)
    if cc[0] < 10:
        return f"cute-dsl backend (MLA decode kernel) requires SM100+, got SM{cc[0]}{cc[1]}"
    if isinstance(bmm1_scale, torch.Tensor):
        return (
            "cute-dsl backend (MLA decode kernel) does not support tensor bmm1_scale, "
            "please pass a float value"
        )
    if isinstance(bmm2_scale, torch.Tensor):
        return (
            "cute-dsl backend (MLA decode kernel) does not support tensor bmm2_scale, "
            "please pass a float value"
        )
    # Cute-dsl supports sinks via its modular variant (auto-promoted when
    # sinks is set, per flashinfer.cute_dsl.attention.mla_dispatch). The
    # public sinks contract is a single per-head tensor or a length-1 list;
    # reject the ambiguous len>1 case here so the autotune dispatcher and
    # the explicit backend="cute-dsl" path see the same error.
    if isinstance(sinks, (list, tuple)) and len(sinks) != 1:
        return (
            f"cute-dsl backend (MLA decode kernel) expects sinks to be a "
            f"single tensor or a length-1 list/tuple; got len={len(sinks)}"
        )
    if sparse_mla_top_k > 0:
        return "cute-dsl backend (MLA decode kernel) does not support sparse_mla_top_k"
    if skip_softmax_threshold_scale_factor is not None:
        return (
            "cute-dsl backend (MLA decode kernel) does not support "
            "skip_softmax_threshold_scale_factor"
        )
    if not uses_shared_paged_kv_idx:
        return (
            "cute-dsl backend (MLA decode kernel) does not support separate KV "
            "page indices (uses_shared_paged_kv_idx=False)"
        )
    # LSE is supported on the monolithic path; the modular path raises a
    # clear NotImplementedError in wrappers/batch_mla.py if it gets picked
    # for an LSE request (e.g. when ``sinks`` forces the modular dispatch).
    # We don't pre-reject here so that the common case
    # (cute_dsl_impl="auto" + LSE + no sinks → monolithic) goes through.

    _, q_len, num_heads, _ = query.shape
    try:
        from ..cute_dsl.attention.mla_dispatch import _resolve_impl

        resolved_impl = _resolve_impl(requested=cute_dsl_impl, kwargs={"sinks": sinks})
    except (ValueError, ImportError) as e:
        return f"cute-dsl backend (MLA decode kernel): {e}"

    try:
        if resolved_impl == "monolithic":
            from ..cute_dsl.attention.monolithic.mla_decode import _check_can_implement
        else:
            from ..cute_dsl.attention.wrappers.batch_mla import _check_can_implement

        _check_can_implement(
            torch_dtype=query.dtype,
            torch_out_dtype=out_dtype,
            page_size=page_size,
            num_heads=num_heads,
            seq_len_q=q_len,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            is_persistent=not is_var_seq,
            is_var_seq=is_var_seq,
            is_var_split_kv=False,
        )
    except (ValueError, ImportError) as e:
        return f"cute-dsl backend (MLA decode kernel) cannot implement this configuration: {e}"
    return None


def _build_mla_decode_tuning_config(
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    workspace_buffer: torch.Tensor,
    runner_names: List[str],
    q_len: int,
    num_heads: int,
    kv_lora_rank: int,
    max_seq_len: int,
    device: torch.device,
):
    """Per-call TuningConfig with bucket-capped batch sweep and closure-based initializers.

    The DynamicTensorSpec sweeps batch dim across all four ``inputs`` tensors
    (query, block_tables, seq_lens, out). ``block_tables`` is initialized via
    ``random_(0, num_pages)`` which wraps mod kv_cache size — safe for autotune
    profiling because MLA decode reads kv_cache and never writes it, so aliased
    page reads give correct timing measurements. ``seq_lens`` is filled
    homogeneously with ``min(max_seq_len, provisioned_max_seq_len)``.
    """
    from ..autotuner import DynamicTensorSpec, TuningConfig, make_bucket_mapper

    # kv_cache may be 3D [num_pages, page_size, D] or 4D
    # [num_pages, 1, page_size, D] after `_check_trtllm_gen_mla_shape` —
    # page_size is the second-to-last dim in both layouts.
    page_size = kv_cache.shape[-2]
    provisioned_max_seq_len = block_tables.shape[-1] * page_size
    profile_seq_len = min(max_seq_len, provisioned_max_seq_len)
    num_pages = kv_cache.shape[0]

    buckets = _compute_mla_decode_buckets(
        workspace_buffer,
        runner_names,
        q_len,
        num_heads,
        kv_lora_rank,
        device,
    )

    def init_block_tables(shapes, dtype, device):
        tensor = torch.empty(shapes, dtype=dtype, device=device)
        tensor.random_(0, num_pages)
        return tensor

    def init_seq_lens(shapes, dtype, device):
        tensor = torch.empty(shapes, dtype=dtype, device=device)
        tensor.fill_(profile_seq_len)
        return tensor

    return TuningConfig(
        dynamic_tensor_specs=(
            DynamicTensorSpec(
                input_idx=(0, 1, 2, 3),
                dim_idx=(0, 0, 0, 0),
                gen_tuning_buckets=buckets,
                map_to_tuning_buckets=make_bucket_mapper(buckets, round_map=False),
                tensor_initializers=[None, init_block_tables, init_seq_lens, None],
            ),
        ),
        use_cuda_graph=True,
        use_cold_l2_cache=True,
    )


class TrtllmGenMlaDecodeRunner(TunableRunner):
    """Wraps ``trtllm_paged_attention_decode`` for the autotuner.

    Non-tensor parameters of ``trtllm_batch_decode_with_kv_cache_mla`` are
    stashed at construction time. The autotuner passes
    ``[query, block_tables, seq_lens, out]`` via ``inputs`` and reads the
    pre-normalized ``kv_cache`` and ``workspace_buffer`` from ``self``.

    The dispatcher is responsible for: (a) normalizing kv_cache via
    ``_check_trtllm_gen_mla_shape``, (b) computing ``sm_count``, and (c)
    applying the ``bmm*_scale * log2e`` conversion for the tensor case
    BEFORE constructing this runner.
    """

    def __init__(
        self,
        *,
        kv_cache: torch.Tensor,
        workspace_buffer: torch.Tensor,
        sm_count: int,
        qk_nope_head_dim: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        max_seq_len: int,
        sparse_mla_top_k: int,
        bmm1_scale,
        bmm2_scale,
        sinks: Optional[List[torch.Tensor]],
        skip_softmax_threshold_scale_factor: Optional[float],
        enable_pdl: bool,
        is_var_seq: bool,
        uses_shared_paged_kv_idx: bool,
        return_lse: bool,
        lse: Optional[torch.Tensor],
    ):
        self._run = get_trtllm_gen_fmha_module().trtllm_paged_attention_decode
        self.kv_cache = kv_cache
        self.workspace_buffer = workspace_buffer
        self.sm_count = sm_count
        # Allocated lazily for autotune profiling and reused (grown if a later
        # profile needs more). The final request may instead pass a caller-owned
        # buffer directly to forward(). The kernel self-resets the counters after
        # each ordered launch, so neither path re-zeros between launches.
        self._multi_ctas_kv_counter_buffer: Optional[torch.Tensor] = None
        self.qk_nope_head_dim = qk_nope_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        # kv_cache has been normalized to 4D by _check_trtllm_gen_mla_shape;
        # the page_size dim is -2 in both 3D (legacy) and 4D shapes.
        self.page_size = kv_cache.shape[-2]
        self.max_seq_len = max_seq_len
        self.sparse_mla_top_k = sparse_mla_top_k
        self.bmm1_scale = bmm1_scale
        self.bmm2_scale = bmm2_scale
        self.sinks = sinks
        self.skip_softmax_threshold_scale_factor = skip_softmax_threshold_scale_factor
        self.enable_pdl = enable_pdl
        self.is_var_seq = is_var_seq
        self.uses_shared_paged_kv_idx = uses_shared_paged_kv_idx
        self.return_lse = return_lse
        self.lse = lse

    def __hash__(self):
        # The default `TunableRunner.__hash__` walks `self.__dict__` and falls
        # back to `id(...)` for unhashable values; our kv_cache / workspace /
        # sinks attributes are tensors or lists whose `id()` differs per
        # dispatcher call, which would poison the autotune cache key. All
        # tactic-determining state is already captured by
        # `get_cache_key_extras`, so return a class-stable hash here.
        return hash(type(self))

    def get_valid_tactics(self, inputs, profile) -> List[int]:
        return [-1]

    def get_cache_key_extras(self, inputs):
        q, _, _, out = inputs
        sinks_key = (
            None
            if self.sinks is None
            else tuple((tuple(t.shape), t.dtype) for t in self.sinks)
        )
        return (
            q.dtype,
            self.kv_cache.dtype,
            out.dtype,
            self.qk_nope_head_dim,
            self.kv_lora_rank,
            self.qk_rope_head_dim,
            self.page_size,
            _round_to_seq_len_bucket(self.max_seq_len),
            self.sparse_mla_top_k,
            self.is_var_seq,
            self.uses_shared_paged_kv_idx,
            self.enable_pdl,
            "bmm1_tensor"
            if isinstance(self.bmm1_scale, torch.Tensor)
            else "bmm1_float",
            "bmm2_tensor"
            if isinstance(self.bmm2_scale, torch.Tensor)
            else "bmm2_float",
            sinks_key,
            self.skip_softmax_threshold_scale_factor,
            self.return_lse,
        )

    def forward(
        self,
        inputs,
        tactic: int = -1,
        do_preparation: bool = False,
        multi_ctas_kv_counter_buffer: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        query, block_tables, seq_lens, out = inputs
        batch_size = query.size(0)
        max_q_len = query.size(1)
        num_qo_heads = query.size(2)
        query_flat = query.flatten(0, 1)

        if self.return_lse:
            lse_shape = (batch_size * max_q_len, num_qo_heads)
            # Reuse caller's lse when its shape matches the current input
            # (final dispatcher call); otherwise allocate fresh for the
            # autotune profile loop (synthetic inputs at bucket batch dims).
            if self.lse is not None and tuple(self.lse.shape) == lse_shape:
                lse = self.lse
            else:
                lse = torch.empty(lse_shape, dtype=torch.float32, device=query.device)
            lse_stride_tokens = lse.stride(0)
            lse_stride_heads = lse.stride(1)
        else:
            lse = None
            lse_stride_tokens = 0
            lse_stride_heads = 0

        counter_buffer = multi_ctas_kv_counter_buffer
        if counter_buffer is None:
            counter_buffer = self._multi_ctas_kv_counter_buffer
            required_counter_bytes = get_trtllm_gen_multi_ctas_kv_counter_bytes(
                batch_size, num_qo_heads, self.sm_count
            )
            counter_buffer_bytes = (
                0
                if counter_buffer is None
                else counter_buffer.numel() * counter_buffer.element_size()
            )
            if counter_buffer is None or counter_buffer_bytes < required_counter_bytes:
                counter_buffer = _get_trtllm_gen_multi_ctas_kv_counter_buffer(
                    batch_size,
                    num_qo_heads,
                    self.sm_count,
                    query.device,
                )
                self._multi_ctas_kv_counter_buffer = counter_buffer
        multi_ctas_kv_counter_buffer = counter_buffer
        self._run(
            out,
            None,  # fp4 output (unsupported by wrapper)
            query_flat,
            self.kv_cache,
            self.kv_cache,  # kv passed twice (K/V views over the same buffer)
            self.workspace_buffer,
            multi_ctas_kv_counter_buffer,
            block_tables,
            seq_lens,
            max_q_len,
            self.max_seq_len,
            self.bmm1_scale,
            self.bmm2_scale,
            -1,  # o_sf_scale
            -1,  # o_sf_vec_size
            0,  # o_sf_start_index
            batch_size,
            -1,  # window_left
            self.sparse_mla_top_k,
            self.sm_count,
            self.enable_pdl,
            self.workspace_buffer.numel() * self.workspace_buffer.element_size(),
            self.sinks,
            None,  # cum_seq_lens_q
            None,  # key_block_scales
            None,  # value_block_scales
            self.skip_softmax_threshold_scale_factor,
            self.uses_shared_paged_kv_idx,
            lse,
            lse_stride_tokens,
            lse_stride_heads,
        )
        return out


class CuteDslMlaDecodeRunner(TunableRunner):
    """Wraps ``cute_dsl_mla_decode`` for the autotuner.

    Only accepts the cute-dsl-compatible parameter subset; the dispatcher's
    auto path filters out incompatible configurations via
    ``_cute_dsl_incompatibility_reason`` before constructing this runner.

    Note: cute-dsl uses ``softmax_scale``/``output_scale`` as its parameter
    names for the bmm1/bmm2 scales. Both are floats only (tensor form is
    unsupported and filtered out upstream).
    """

    def __init__(
        self,
        *,
        kv_cache: torch.Tensor,
        workspace_buffer: torch.Tensor,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        max_seq_len: int,
        softmax_scale: float,
        output_scale: float,
        out_dtype: torch.dtype,
        enable_pdl: bool,
        is_var_seq: bool,
        uses_shared_paged_kv_idx: bool,
        lse: Optional[torch.Tensor],
        return_lse: bool,
        sinks: Optional[torch.Tensor],
        cute_dsl_impl: str,
    ):
        from ..cute_dsl.attention import cute_dsl_mla_decode

        self._run = cute_dsl_mla_decode
        self.kv_cache = kv_cache
        self.workspace_buffer = workspace_buffer
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        # kv_cache may be 3D [num_pages, page_size, D] or 4D
        # [num_pages, 1, page_size, D] after `_check_trtllm_gen_mla_shape` at
        # dispatcher level — page_size is the second-to-last dim in both.
        self.page_size = kv_cache.shape[-2]
        self.max_seq_len = max_seq_len
        self.softmax_scale = softmax_scale
        self.output_scale = output_scale
        self.out_dtype = out_dtype
        self.enable_pdl = enable_pdl
        self.is_var_seq = is_var_seq
        self.uses_shared_paged_kv_idx = uses_shared_paged_kv_idx
        self.lse = lse
        self.return_lse = return_lse
        self.sinks = sinks
        self.cute_dsl_impl = cute_dsl_impl

    def __hash__(self):
        # See TrtllmGenMlaDecodeRunner.__hash__ — tactic-determining state is
        # captured by get_cache_key_extras; the default per-instance hash
        # would key on `id(kv_cache)` / `id(workspace_buffer)` and break the
        # autotune cache across dispatcher calls.
        return hash(type(self))

    def get_valid_tactics(self, inputs, profile) -> List[int]:
        # Workspace-bound: cute-dsl's per-CTA split-K state grows with B.
        # If the caller's workspace can't fit batch=B for this profile, opt
        # out so the autotuner skips us (no JIT cost) and trtllm-gen wins by
        # default for that bucket.
        from ..cute_dsl.attention.wrappers.batch_mla import (
            _get_split_kv_and_workspace_size,
        )
        from ..cute_dsl.utils import get_num_sm

        q = inputs[0]
        B, q_len, num_heads, _ = q.shape
        _, ws = _get_split_kv_and_workspace_size(
            B, q_len, num_heads, self.kv_lora_rank, get_num_sm(q.device)
        )
        workspace_bytes = (
            self.workspace_buffer.numel() * self.workspace_buffer.element_size()
        )
        if ws > workspace_bytes:
            return []
        return [-1]

    def get_cache_key_extras(self, inputs):
        q, _, _, out = inputs
        # Cute-dsl rejects sparse/skip-softmax/tensor-scales upstream, so
        # those are omitted from extras as constants for this runner.
        # ``sinks`` and ``cute_dsl_impl`` are included because they flip the
        # impl (modular vs monolithic) inside ``cute_dsl_mla_decode``.
        sinks_key = (
            None if self.sinks is None else (tuple(self.sinks.shape), self.sinks.dtype)
        )
        return (
            q.dtype,
            self.kv_cache.dtype,
            out.dtype,
            self.qk_nope_head_dim,
            self.kv_lora_rank,
            self.qk_rope_head_dim,
            self.page_size,
            _round_to_seq_len_bucket(self.max_seq_len),
            self.is_var_seq,
            self.uses_shared_paged_kv_idx,
            self.enable_pdl,
            sinks_key,
            self.cute_dsl_impl,
        )

    def forward(
        self,
        inputs,
        tactic: int = -1,
        do_preparation: bool = False,
        **kwargs,
    ):
        query, block_tables, seq_lens, out = inputs
        return self._run(
            query=query,
            kv_cache=self.kv_cache,
            workspace_buffer=self.workspace_buffer,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=self.max_seq_len,
            softmax_scale=self.softmax_scale,
            output_scale=self.output_scale,
            out=out,
            out_dtype=self.out_dtype,
            is_var_seq=self.is_var_seq,
            enable_pdl=self.enable_pdl,
            lse=self.lse,
            return_lse=self.return_lse,
            sinks=self.sinks,
            cute_dsl_impl=self.cute_dsl_impl,
        )


@flashinfer_api(trace=trtllm_batch_decode_mla_trace_dispatch)
def trtllm_batch_decode_with_kv_cache_mla(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    workspace_buffer: torch.Tensor,
    qk_nope_head_dim: int,  # TODO: remove in 1.0?
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
    r"""Decode MLA with TRTLLM-GEN, CuteDSL, XQA, or SM120/SM121 sparse kernels.

    With ``backend="auto"``, SM100/SM103 devices use TRTLLM-GEN for sparse MLA
    when ``sparse_mla_top_k > 0``. SM120/SM121 devices use the packed sparse
    backend for ``sparse_mla_top_k > 0`` and XQA for dense decode.

    Parameters
    ----------
    query : torch.Tensor
        Query tensor with shape
        ``[batch_size, q_len_per_request, num_heads, head_dim_qk]`` where
        ``head_dim_qk = kv_lora_rank + qk_rope_head_dim``. For the SM120/SM121
        v32/GLM sparse backend, this must be BF16 with ``head_dim_qk == 576``.
    kv_cache : torch.Tensor
        For TRTLLM-GEN, CuteDSL, and XQA, the paged KV cache is
        ``[num_pages, page_size, kv_lora_rank + qk_rope_head_dim]`` or
        ``[num_pages, 1, page_size, kv_lora_rank + qk_rope_head_dim]`` and uses
        the query-compatible dense dtype. For the SM120/SM121 v32/GLM sparse
        backend, this is a packed uint8 cache with 656 bytes per token, shaped
        ``[num_pages, page_size, 656]`` or ``[num_pages, 1, page_size, 656]``.
    workspace_buffer : torch.Tensor
        Pre-allocated workspace buffer. Must be zero-initialized on first use
        by kernels that use semaphore state.
    qk_nope_head_dim : int
        Non-RoPE query dimension. Dense MLA paths commonly use ``128`` or
        ``64`` depending on model. The SM120/SM121 sparse v32/GLM backend
        ignores this value and validates ``query.shape[-1] == 576`` instead.
    kv_lora_rank : int
        Latent KV rank. TRTLLM-GEN and SM120/SM121 sparse v32/GLM use ``512``.
    qk_rope_head_dim : int
        RoPE head dimension. Sparse MLA paths use ``64``.
    block_tables : torch.Tensor
        Page table for dense MLA backends when ``sparse_mla_top_k == 0``. For
        SM100/SM103 TRTLLM-GEN sparse MLA it is the usual paged block table.
        When ``cum_seq_lens_q`` is provided with sparse MLA, pass compact
        sparse rows in flattened query-token order with shape
        ``[total_q, sparse_mla_top_k]``.
        For SM120/SM121 sparse v32/GLM, it is the sparse index matrix and must
        have shape ``[batch_size, q_len_per_request, sparse_mla_top_k]`` with
        int32 physical token indices.
        With ``backend="trtllm-gen"``, the final dimension may use its native
        width and does not need padding to a multiple of ``128 / page_size``.
    seq_lens : Optional[torch.Tensor]
        Per-request KV sequence lengths for dense and TRTLLM-GEN paths. For
        SM120/SM121 sparse v32/GLM, pass ``[batch_size, q_len_per_request]`` or
        flattened ``[batch_size * q_len_per_request]`` active top-k lengths; if
        ``None``, every column in ``block_tables`` is active.
    max_seq_len : int
        Maximum KV sequence length used for dense/TRTLLM-GEN scheduling.
        Ignored by the SM120/SM121 sparse v32/GLM backend.
    sparse_mla_top_k : int
        Enables sparse MLA when greater than zero. On SM100/SM103 this selects
        the TRTLLM-GEN sparse page-table path. On SM120/SM121 with
        ``backend="auto"`` or ``backend="sparse"``, this is the width of the
        packed v32/GLM sparse index matrix. The TRTLLM-GEN backend supports
        dense query input or flattened query input plus ``cum_seq_lens_q``.
    out : Optional[torch.Tensor]
        Output tensor. If not provided, it is allocated internally.
    bmm1_scale : Union[float, torch.Tensor]
        Fused scale for MLA BMM1. TRTLLM-GEN accepts a FP32 tensor or float.
        CuteDSL, XQA, and SM120/SM121 sparse v32/GLM require a float.
    bmm2_scale : Union[float, torch.Tensor]
        Fused scale for MLA BMM2. TRTLLM-GEN accepts a FP32 tensor or float.
        CuteDSL and XQA require a float. SM120/SM121 sparse v32/GLM requires
        ``1.0``.
    sinks : Optional[List[torch.Tensor]]
        Additional value per head in the denominator of the softmax.
        Supported by ``trtllm-gen``, ``cute-dsl``, and ``sparse``.
        On ``cute-dsl`` this requires the modular implementation;
        ``cute_dsl_impl="auto"`` (the default) promotes to modular
        automatically, and ``cute_dsl_impl="monolithic"`` with sinks set raises
        :class:`ValueError`.
    skip_softmax_threshold_scale_factor: threshold scale factor for skipping softmax operations.
        Providing a value for this parameter enables skip-softmax sparsity as described in: https://arxiv.org/abs/2512.12087
        If no value is provided, then standard attention is used.
        Setting the threshold to a higher value generally increases kernel performance at the cost of accuracy degradation.
        The actual threshold value equals the provided threshold_scale_factor divided by the context length.
    enable_pdl : Optional[bool]
        Programmatic Dependent Launch toggle.  When ``None`` (default), auto-detects
        support from the query device.  Honoured by the ``trtllm-gen`` and ``xqa``
        backends; ignored by ``cute-dsl``.
    backend : str = "auto"
        Implementation backend. Valid values are ``"auto"``, ``"xqa"``,
        ``"trtllm-gen"``, ``"cute-dsl"``, and ``"sparse"``. ``"auto"``
        chooses ``"trtllm-gen"`` for SM100/SM103 sparse MLA and chooses
        ``"sparse"`` for SM120/SM121 when ``sparse_mla_top_k > 0``; otherwise
        SM120/SM121 dense decode uses ``"xqa"``.
        The ``cute-dsl`` backend has two interchangeable implementations
        (``monolithic`` and ``modular``) on the same shape/dtype envelope;
        which one runs is controlled by the ``cute_dsl_impl`` kwarg below.
    is_var_seq : bool
        Whether the sequence length is variable.
        If True, the sequence length is variable.
        Otherwise,the sequence length is fixed for all the requests in the batch.
    uses_shared_paged_kv_idx : bool = True
        Whether K and V page indices are shared as a unified index.
        True (default) uses vLLM/FlashInfer layout with a 2D page table.
        False uses TRT-LLM layout with a 3D page table ``[batch_size, 2, max_num_pages_per_seq]``.
        False is only supported by TRTLLM-GEN.
    lse : Optional[torch.Tensor] = None
        Optional pre-allocated buffer for Log-Sum-Exp values. Supported by
        ``trtllm-gen``, ``cute-dsl``, and ``sparse`` backends. Must have
        dtype ``torch.float32``. Accepted shapes:

        * ``[batch_size * q_len_per_request, num_qo_heads]`` (TRTLLM-GEN
          native; accepted by sparse), or
        * ``[batch_size, q_len_per_request, num_qo_heads]`` (cute-dsl native;
          also accepted by cute-dsl).

        If ``return_lse`` is True and this is None, a buffer will be
        allocated by the backend.
    return_lse : bool = False
        Whether to return LSE values. Supported by ``trtllm-gen``,
        ``cute-dsl``, and ``sparse`` backends. When True, the function
        returns ``(out, lse)``.
    cute_dsl_impl : str = "auto"
        Which cute-dsl implementation to use. Honored when
        ``backend="cute-dsl"`` and when ``backend="auto"`` considers the
        cute-dsl candidate; ignored for non-cute-dsl backends.

        * ``"auto"`` (default) — picks monolithic by default, automatically
          promoted to modular when the call uses a feature monolithic
          doesn't support (currently ``sinks``).
        * ``"modular"`` — strict.  Always run the modular kernels.
        * ``"monolithic"`` — strict.  Always run the monolithic kernels;
          raise :class:`ValueError` if the call uses any modular-only
          feature (e.g. ``sinks``).
    kv_scale_format : str = "auto"
        Scale semantics for the SM120/SM121 packed v32/GLM sparse backend.
        ``"auto"`` and ``"pow2_fp32"`` select DSv3.2 power-of-2 FP32 inline
        scales; ``"arbitrary_fp32"`` selects GLM-style arbitrary FP32 inline scales.
        Ignored by the ``trtllm-gen``, ``xqa``, and ``cute-dsl`` backends.
    cum_seq_lens_q : Optional[torch.Tensor] = None
        Cumulative query sequence lengths for variable-length query support,
        shape ``[batch_size + 1]``, dtype ``torch.int32``. Must be a 1D tensor
        with at least two entries. When ``max_q_len`` is not provided, this
        function validates that it starts with 0, ends at ``query.size(0)``,
        and is monotonically non-decreasing. Only supported by the
        ``trtllm-gen`` backend. When provided, ``query`` must have shape
        ``[total_q, num_heads, head_dim_qk]``.
        For best performance, provide ``max_q_len`` together with
        ``cum_seq_lens_q`` to avoid host-side metadata validation.
    max_q_len : Optional[int] = None
        Maximum query sequence length across all requests when using
        ``cum_seq_lens_q``. Provide with ``cum_seq_lens_q`` to avoid
        host-side metadata validation. Must be greater than or equal to the
        maximum segment length represented by ``cum_seq_lens_q``. Over-estimation
        is safe but may waste work; under-estimation is invalid and may produce
        incorrect output.
    multi_ctas_kv_counter_buffer : Optional[torch.Tensor] = None
        Optional caller-owned counter buffer for the ``trtllm-gen`` backend.
        It must be contiguous, remain alive for every launch or CUDA graph replay
        that uses it, and be zero-initialized once. Allocate at least the number
        of bytes returned by ``get_trtllm_gen_multi_ctas_kv_counter_bytes`` for
        the current batch size, query-head count, and device SM count; a contiguous
        ``torch.uint8`` tensor created with ``torch.zeros`` is recommended. Reuse
        is safe only for ordered, non-overlapping launches; use a distinct buffer
        for each concurrently executing CUDA stream or graph. Autotune profiling
        uses runner-owned internal storage; the caller buffer is used only for the
        final request.

    Note
    ----
    In MLA, the actual BMM1 and BMM2 scales applied would be fused as:
    bmm1_scale = q_scale * k_scale * sm_scale / (head_dim_qk ** 0.5)
    bmm2_scale = v_scale * o_scale
    or,
    bmm1_scale = torch.Tensor([q_scale * k_scale * sm_scale / (head_dim_qk ** 0.5))
    bmm2_scale = torch.Tensor([v_scale * o_scale])

    The two scale factors should be static constant for cuda graph capture.
    Either (bmm1_scale, bmm2_scale) or (bmm1_scale_log2_tensor, bmm2_scale_tensor) should be provided.

    For static constant scale factors, the scale factors should be provided as float.
        - (bmm1_scale, bmm2_scale)
    For on-device fused scale tensors, which could dynamically change, the scale factors should be provided as torch.Tensor.
        - (bmm1_scale_log2_tensor, bmm2_scale_tensor)
        - Currently, only fp8 tensor core operation supports this mode.
    When both are provided, the dynamic scale factor tensors will be used.

    Autotune
    --------
    On SM100/SM103 dense MLA, calling under ``flashinfer.autotune(True)`` with
    ``backend="auto"`` profiles both ``trtllm-gen`` and ``cute-dsl`` across a
    bucketed batch sweep up to each runner's kernel/workspace cap and caches the
    winning runner per shape signature. Subsequent calls under
    ``autotune(False)`` dispatch to the cached choice; any batch outside the
    tuned range falls back to a default runner with a one-time warning.

    The autotune bucket range and cache key do **not** depend on
    ``kv_cache.shape[0]`` (the number of pages in the pool), so reallocating the
    pool between tuning and inference does not invalidate cached choices. However,
    the **page-aliasing ratio** during profiling does depend on the pool size:
    synthetic ``block_tables`` are filled by uniform random sampling into
    ``[0, kv_cache.shape[0])``, so a small pool produces high aliasing
    (L2-resident reads) and a large pool produces low aliasing (HBM-bound reads).
    For best profile fidelity, autotune with a ``kv_cache`` whose size reflects
    the production page-sharing pattern of your workload (e.g., heavily shared
    prefix → smaller pool; independent contexts → larger pool).
    """
    if isinstance(bmm1_scale, torch.Tensor):
        if bmm1_scale.dtype != torch.float32:
            raise TypeError("bmm1_scale tensor must have dtype torch.float32")
    if isinstance(bmm2_scale, torch.Tensor):
        if bmm2_scale.dtype != torch.float32:
            raise TypeError("bmm2_scale tensor must have dtype torch.float32")
    if max_q_len is not None and cum_seq_lens_q is None:
        raise ValueError("max_q_len is only supported when cum_seq_lens_q is provided")

    if backend == "auto":
        cc = get_compute_capability(query.device)
        if cc[0] == 12 and sparse_mla_top_k > 0:
            backend = "sparse"
        elif cc[0] != 10:
            backend = "xqa"

    if backend == "xqa":
        if multi_ctas_kv_counter_buffer is not None:
            raise ValueError(
                "multi_ctas_kv_counter_buffer is only supported by the trtllm-gen backend"
            )
        if seq_lens is None:
            raise ValueError("seq_lens is required for XQA MLA")
        if sparse_mla_top_k > 0:
            raise ValueError("XQA MLA does not support sparse_mla_top_k")
        if cum_seq_lens_q is not None or max_q_len is not None:
            raise ValueError("XQA MLA does not support cum_seq_lens_q / max_q_len")
        if not is_sm12x_supported(query.device):
            raise ValueError(
                "XQA MLA requires SM120a (CUDA >= 12.8) or SM121a (CUDA >= 13.0)"
            )
        fp8_ok = (
            query.dtype == torch.float8_e4m3fn and kv_cache.dtype == torch.float8_e4m3fn
        )
        bf16_ok = query.dtype == torch.bfloat16 and kv_cache.dtype == torch.bfloat16
        if not (fp8_ok or bf16_ok):
            raise ValueError(
                f"XQA MLA on SM120/SM121 supports (fp8, fp8) or (bfloat16, bfloat16) only, got {query.dtype} and {kv_cache.dtype}"
            )
        if sinks is not None:
            raise ValueError("XQA MLA does not support sinks")
        if query.size(1) != 1:
            raise ValueError(
                f"XQA MLA only supports q_len_per_request == 1, got {query.size(1)}"
            )
        if skip_softmax_threshold_scale_factor is not None:
            raise ValueError("skip_softmax is not supported for XQA backend")
        if not uses_shared_paged_kv_idx:
            raise ValueError(
                "XQA MLA does not support separate KV page indices (uses_shared_paged_kv_idx=False)"
            )
        if return_lse or lse is not None:
            raise NotImplementedError(
                "XQA MLA backend does not support return_lse/lse output"
            )
        return xqa_batch_decode_with_kv_cache_mla(
            query,
            kv_cache,
            workspace_buffer,
            -1,  # Unused, marked for removal.
            kv_lora_rank,
            qk_rope_head_dim,
            block_tables,
            seq_lens,
            max_seq_len,
            out,
            bmm1_scale,
            bmm2_scale,
            sinks,
            enable_pdl,
        )
    if backend not in ("auto", "trtllm-gen", "cute-dsl", "sparse"):
        raise ValueError(f"Backend {backend} not supported")

    if backend == "sparse":
        if multi_ctas_kv_counter_buffer is not None:
            raise ValueError(
                "multi_ctas_kv_counter_buffer is only supported by the trtllm-gen backend"
            )
        return _trtllm_batch_decode_sparse_mla_v32_sm120(
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

    if seq_lens is None:
        raise ValueError(
            "seq_lens is required for trtllm-gen and cute-dsl MLA backends"
        )

    # log2e fusion is a trtllm-gen-only transform (the kernel expects
    # log2-form scales for the tensor case). Apply after the xqa branch so
    # that calling this function with backend="xqa" and calling
    # xqa_batch_decode_with_kv_cache_mla directly yield the same kernel input.
    if isinstance(bmm1_scale, torch.Tensor):
        bmm1_scale = bmm1_scale * log2e

    # Shared setup for the trtllm-gen / cute-dsl autotune dispatch.
    enable_pdl = device_support_pdl(query.device) if enable_pdl is None else enable_pdl
    sm_count = get_device_sm_count(query.device)

    block_size = kv_cache.size(-2)
    trtllm_gen_not_supported_reason: Optional[str] = None
    if block_size != 32 and block_size != 64:
        trtllm_gen_not_supported_reason = (
            f"trtllm-gen requires block_size in (32, 64), got {block_size}"
        )

    if skip_softmax_threshold_scale_factor is not None and sparse_mla_top_k != 0:
        raise ValueError("skip_softmax is not supported for sparse MLA")

    has_var_q = cum_seq_lens_q is not None
    if has_var_q:
        if backend == "cute-dsl":
            raise ValueError("cute-dsl MLA does not support cum_seq_lens_q")
        if return_lse or lse is not None:
            raise NotImplementedError(
                "trtllm-gen MLA does not support return_lse/lse with cum_seq_lens_q"
            )
        if query.ndim != 3:
            raise ValueError(
                "query must have shape [total_q, num_heads, head_dim_qk] "
                "when cum_seq_lens_q is provided"
            )
        check_shape_dtype_device(
            cum_seq_lens_q,
            None,
            torch.int32,
            query.device,
            "cum_seq_lens_q",
        )
        if cum_seq_lens_q.ndim != 1:
            raise ValueError(
                f"Expected cum_seq_lens_q.ndim == 1, got {cum_seq_lens_q.ndim}"
            )
        if cum_seq_lens_q.size(0) < 2:
            raise ValueError("cum_seq_lens_q must contain at least two entries")
        batch_size = cum_seq_lens_q.size(0) - 1
        if batch_size != seq_lens.size(0):
            raise ValueError(
                "Batch size mismatch: cum_seq_lens_q describes "
                f"{batch_size} sequences, but seq_lens has {seq_lens.size(0)} entries"
            )
        if max_q_len is None:
            cum_seq_lens_q_host = cum_seq_lens_q.cpu()
            if cum_seq_lens_q_host[0].item() != 0:
                raise ValueError("cum_seq_lens_q must start with 0")
            if cum_seq_lens_q_host[-1].item() != query.size(0):
                raise ValueError(
                    "cum_seq_lens_q[-1] must match the flattened query length"
                )
            q_lens = cum_seq_lens_q_host[1:] - cum_seq_lens_q_host[:-1]
            if torch.any(q_lens < 0).item():
                raise ValueError("cum_seq_lens_q must be monotonically non-decreasing")
            max_q_len = q_lens.max().item()
            if max_q_len <= 0:
                raise ValueError(
                    "cum_seq_lens_q must describe at least one query token"
                )
        elif max_q_len <= 0:
            raise ValueError("max_q_len must be greater than 0")
        elif max_q_len > query.size(0):
            raise ValueError("max_q_len cannot exceed the flattened query length")

        kv_cache = _check_trtllm_gen_mla_shape(
            query,
            kv_cache,
            kv_lora_rank,
            qk_rope_head_dim,
            sparse_mla_top_k,
            block_tables,
            block_size,
            uses_shared_paged_kv_idx,
            batch_size=batch_size,
            max_q_len=max_q_len,
            require_aligned_block_table=False,
        )

        expected_out_shape = query.shape[:-1] + (kv_lora_rank,)
        if out is None:
            out = torch.empty(
                expected_out_shape, dtype=torch.bfloat16, device=query.device
            )
        else:
            check_shape_dtype_device(
                out,
                expected_out_shape,
                torch.bfloat16,
                query.device,
                "out",
            )

        multi_ctas_kv_counter_buffer = _resolve_trtllm_gen_multi_ctas_kv_counter_buffer(
            multi_ctas_kv_counter_buffer,
            batch_size,
            query.size(1),
            sm_count,
            query.device,
        )
        get_trtllm_gen_fmha_module().trtllm_paged_attention_decode(
            out,
            None,  # fp4 output (unsupported by wrapper)
            query,
            kv_cache,
            kv_cache,
            workspace_buffer,
            multi_ctas_kv_counter_buffer,
            block_tables,
            seq_lens,
            max_q_len,
            max_seq_len,
            bmm1_scale,
            bmm2_scale,
            -1,  # o_sf_scale
            -1,  # o_sf_vec_size
            0,  # o_sf_start_index
            batch_size,
            -1,  # window_left
            sparse_mla_top_k,
            sm_count,
            enable_pdl,
            workspace_buffer.numel() * workspace_buffer.element_size(),
            sinks,
            cum_seq_lens_q,
            None,  # key_block_scales
            None,  # value_block_scales
            skip_softmax_threshold_scale_factor,
            uses_shared_paged_kv_idx,
            None,  # lse
            0,  # lse_stride_tokens
            0,  # lse_stride_heads
        )
        return out

    # Normalize kv_cache to 4D and validate MLA dimensions. Despite the name,
    # the shape/dim checks here apply to both backends.
    kv_cache = _check_trtllm_gen_mla_shape(
        query,
        kv_cache,
        kv_lora_rank,
        qk_rope_head_dim,
        sparse_mla_top_k,
        block_tables,
        block_size,
        uses_shared_paged_kv_idx,
        require_aligned_block_table=backend != "trtllm-gen",
    )

    # Pre-allocate `out` so non-swept dims have a template for autotune
    # profiling (the autotuner inherits non-swept dims from caller tensors).
    expected_out_shape = query.shape[:-1] + (kv_lora_rank,)
    if out is None:
        out = torch.empty(expected_out_shape, dtype=torch.bfloat16, device=query.device)
    else:
        check_shape_dtype_device(
            out,
            expected_out_shape,
            torch.bfloat16,
            query.device,
            "out",
        )

    # Remember the caller-supplied lse so we can return it in its original
    # shape: 2D ``(B*q_len, H)`` stays 2D, 3D ``(B, q_len, H)`` stays 3D, and
    # an allocated default stays 2D.  Internally we normalize to 2D for the
    # backend dispatch (matches trtllm-gen's native layout).
    user_lse = lse
    if return_lse:
        flat_lse_shape = (query.size(0) * query.size(1), query.size(2))
        nested_lse_shape = (query.size(0), query.size(1), query.size(2))
        if lse is None:
            lse = torch.empty(flat_lse_shape, dtype=torch.float32, device=query.device)
            user_lse = lse
        elif tuple(lse.shape) == flat_lse_shape:
            check_shape_dtype_device(
                lse, flat_lse_shape, torch.float32, query.device, "lse"
            )
        elif tuple(lse.shape) == nested_lse_shape:
            check_shape_dtype_device(
                lse, nested_lse_shape, torch.float32, query.device, "lse"
            )
            # Normalize to 2D for the backend; .view shares storage so the
            # kernel writes propagate back to user_lse automatically.
            lse = lse.view(flat_lse_shape)
        else:
            raise ValueError(
                f"lse must have shape {flat_lse_shape} or {nested_lse_shape}; "
                f"got {tuple(lse.shape)}"
            )

    page_size = kv_cache.shape[-2]
    cute_dsl_reason = _cute_dsl_incompatibility_reason(
        query,
        out.dtype,
        bmm1_scale,
        bmm2_scale,
        sinks,
        sparse_mla_top_k,
        skip_softmax_threshold_scale_factor,
        uses_shared_paged_kv_idx,
        qk_rope_head_dim,
        kv_lora_rank,
        page_size,
        is_var_seq,
        return_lse,
        lse,
        cute_dsl_impl,
    )
    if backend == "cute-dsl":
        if cute_dsl_reason is not None:
            raise ValueError(cute_dsl_reason)
        runner_names = ["cute-dsl"]
    elif backend == "trtllm-gen":
        if trtllm_gen_not_supported_reason is not None:
            raise ValueError(trtllm_gen_not_supported_reason)
        runner_names = ["trtllm-gen"]
    else:  # backend == "auto"
        runner_names = []
        if trtllm_gen_not_supported_reason is None:
            runner_names.append("trtllm-gen")
        if cute_dsl_reason is None:
            runner_names.append("cute-dsl")
        if not runner_names:
            raise ValueError(
                f"auto: no backend supports this configuration "
                f"(trtllm-gen: {trtllm_gen_not_supported_reason}; "
                f"cute-dsl: {cute_dsl_reason})"
            )

    if multi_ctas_kv_counter_buffer is not None and "trtllm-gen" not in runner_names:
        raise ValueError(
            "multi_ctas_kv_counter_buffer is only supported when a trtllm-gen runner is selected"
        )
    if multi_ctas_kv_counter_buffer is not None:
        multi_ctas_kv_counter_buffer = _resolve_trtllm_gen_multi_ctas_kv_counter_buffer(
            multi_ctas_kv_counter_buffer,
            query.size(0),
            query.size(2),
            sm_count,
            query.device,
        )

    runners: List[TunableRunner] = []
    if "trtllm-gen" in runner_names:
        runners.append(
            TrtllmGenMlaDecodeRunner(
                kv_cache=kv_cache,
                workspace_buffer=workspace_buffer,
                sm_count=sm_count,
                qk_nope_head_dim=qk_nope_head_dim,
                kv_lora_rank=kv_lora_rank,
                qk_rope_head_dim=qk_rope_head_dim,
                max_seq_len=max_seq_len,
                sparse_mla_top_k=sparse_mla_top_k,
                bmm1_scale=bmm1_scale,
                bmm2_scale=bmm2_scale,
                sinks=sinks,
                skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
                enable_pdl=enable_pdl,
                is_var_seq=is_var_seq,
                uses_shared_paged_kv_idx=uses_shared_paged_kv_idx,
                return_lse=return_lse,
                lse=lse,
            )
        )
    if "cute-dsl" in runner_names:
        # Normalize sinks: public contract accepts Optional[List[Tensor]] for
        # legacy reasons, but cute-dsl's modular variant expects a single
        # per-head tensor or None. The list-of-1 case has been guarded by
        # `_cute_dsl_incompatibility_reason` so we can unpack here safely.
        cute_dsl_sinks: Optional[torch.Tensor] = None
        if sinks is not None:
            cute_dsl_sinks = sinks[0] if isinstance(sinks, (list, tuple)) else sinks
        runners.append(
            CuteDslMlaDecodeRunner(
                kv_cache=kv_cache,
                workspace_buffer=workspace_buffer,
                kv_lora_rank=kv_lora_rank,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                max_seq_len=max_seq_len,
                softmax_scale=bmm1_scale,
                output_scale=bmm2_scale,
                out_dtype=out.dtype,
                enable_pdl=enable_pdl,
                is_var_seq=is_var_seq,
                uses_shared_paged_kv_idx=uses_shared_paged_kv_idx,
                lse=lse,
                return_lse=return_lse,
                sinks=cute_dsl_sinks,
                cute_dsl_impl=cute_dsl_impl,
            )
        )

    _, q_len, num_heads, _ = query.shape
    tuning_config = _build_mla_decode_tuning_config(
        kv_cache=kv_cache,
        block_tables=block_tables,
        workspace_buffer=workspace_buffer,
        runner_names=runner_names,
        q_len=q_len,
        num_heads=num_heads,
        kv_lora_rank=kv_lora_rank,
        max_seq_len=max_seq_len,
        device=query.device,
    )
    inputs = [query, block_tables, seq_lens, out]
    runner, tactic = AutoTuner.get().choose_one(
        "trtllm_batch_decode_mla",
        runners,
        tuning_config,
        inputs,
    )
    if isinstance(runner, TrtllmGenMlaDecodeRunner):
        runner(
            inputs=inputs,
            tactic=tactic,
            multi_ctas_kv_counter_buffer=multi_ctas_kv_counter_buffer,
        )
    else:
        runner(inputs=inputs, tactic=tactic)
    if return_lse:
        # Return the lse in the same shape the caller supplied (2D or 3D),
        # or 2D ``(B*q_len, H)`` when we allocated the default.
        return out, user_lse
    return out


@flashinfer_api(trace=xqa_batch_decode_mla_trace)
def xqa_batch_decode_with_kv_cache_mla(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    workspace_buffer: torch.Tensor,
    qk_nope_head_dim: int,  # TODO: remove in 1.0?
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,  # TODO: remove in 1.0?
    out: Optional[torch.Tensor] = None,
    bmm1_scale: Union[float, torch.Tensor] = 1.0,
    bmm2_scale: Union[float, torch.Tensor] = 1.0,
    sinks: Optional[List[torch.Tensor]] = None,
    enable_pdl: bool | None = None,
) -> torch.Tensor:
    r"""XQA-backend batched MLA decode.

    Single-query (MTP-aware) MLA decode kernel optimized for SM120a / SM121a tensor cores.
    Accepts the concatenated ``(q_nope || q_rope)`` query and ``(ckv || kpe)`` paged KV
    cache layout used by DeepSeek-V3 / R1 inference.

    Parameters
    ----------
    query : torch.Tensor
        Query tensor with shape
        ``[batch_size, q_len_per_request, num_heads, head_dim_qk]`` where
        ``head_dim_qk = kv_lora_rank + qk_rope_head_dim``.  Must be the concatenation
        ``[q_nope, q_rope]``.  ``q_len_per_request`` is the MTP query length and is
        currently required to be ``1``.
    kv_cache : torch.Tensor
        Paged KV cache, either 3-D
        ``[num_pages, page_size, kv_lora_rank + qk_rope_head_dim]`` or 4-D
        ``[num_pages, 1, page_size, kv_lora_rank + qk_rope_head_dim]``.  The last
        dimension is the concatenation ``[ckv_cache, kpe_cache]``.  Both shapes are
        accepted for backward compatibility.
    workspace_buffer : torch.Tensor
        Pre-allocated backend scratch workspace buffer.
    qk_nope_head_dim : int
        Non-RoPE head dimension.  Must be ``128``.  Will be removed in 1.0; pass
        ``kv_lora_rank`` instead going forward.
    kv_lora_rank : int
        Rank of the latent KV projection.  Must be ``512``.
    qk_rope_head_dim : int
        RoPE head dimension appended to the latent projection.  Must be ``64``.
    block_tables : torch.Tensor
        Per-request paged KV block table, shape ``[batch_size, num_pages]``.
    seq_lens : torch.Tensor
        Per-request KV sequence length, shape ``[batch_size]``.
    max_seq_len : int
        Maximum KV sequence length used for kernel scheduling.  Will be removed in
        1.0; the kernel reads the per-request lengths from ``seq_lens``.
    out : Optional[torch.Tensor]
        Optional output tensor of shape ``[batch_size, num_heads, kv_lora_rank]``
        and dtype ``torch.bfloat16``.  If ``None``, it is allocated internally.
    bmm1_scale : Union[float, torch.Tensor]
        Fused scale for MLA BMM1 (see Note).  ``float`` for static (CUDA-graph
        safe) scales; ``torch.Tensor`` for on-device dynamic scales (FP8 only).
    bmm2_scale : Union[float, torch.Tensor]
        Fused scale for MLA BMM2 (see Note).  Same typing rules as ``bmm1_scale``.
    sinks : Optional[List[torch.Tensor]]
        Attention-sink tensors.  Currently unsupported and must be ``None``.
    enable_pdl : Optional[bool]
        Programmatic Dependent Launch toggle.  When ``None``, auto-detects support
        from the device.

    Returns
    -------
    torch.Tensor
        Attention output, shape ``[batch_size, num_heads, kv_lora_rank]``, dtype
        ``torch.bfloat16``.

    Note
    ----
    In MLA, the BMM1 and BMM2 scales are fused as:

    .. code-block:: text

        bmm1_scale = q_scale * k_scale * sm_scale / sqrt(head_dim_qk)
        bmm2_scale = v_scale * o_scale

    The scale factors must be static constants for CUDA graph capture.  Either the
    ``(bmm1_scale, bmm2_scale)`` (float) pair or the on-device
    ``(bmm1_scale_log2_tensor, bmm2_scale_tensor)`` tensor pair may be passed.
    When tensor inputs are supplied, the on-device path is taken (FP8 only).
    """
    enable_pdl = device_support_pdl(query.device) if enable_pdl is None else enable_pdl
    sm_count = get_device_sm_count(query.device)

    # Extract block_size (works for both 3D and 4D)
    block_size = kv_cache.size(-2)
    q_len_per_request = query.size(1)
    if q_len_per_request != 1:
        raise ValueError(
            f"XQA MLA only supports q_len_per_request == 1, got {q_len_per_request}"
        )
    if not is_sm12x_supported(query.device):
        raise ValueError(
            "XQA MLA requires SM120a (CUDA >= 12.8) or SM121a (CUDA >= 13.0)"
        )
    fp8_ok = (
        query.dtype == torch.float8_e4m3fn and kv_cache.dtype == torch.float8_e4m3fn
    )
    bf16_ok = query.dtype == torch.bfloat16 and kv_cache.dtype == torch.bfloat16
    if not (fp8_ok or bf16_ok):
        raise ValueError(
            f"XQA MLA supports (fp8, fp8) or (bfloat16, bfloat16) only, got {query.dtype} and {kv_cache.dtype}"
        )
    if sinks is not None:
        raise ValueError("XQA MLA does not support sinks")

    # Validate and normalize to 4D
    kv_cache = _check_trtllm_gen_mla_shape(
        query,
        kv_cache,
        kv_lora_rank,
        qk_rope_head_dim,
        0,  # sparse_mla_top_k
        block_tables,
        block_size,
        True,  # XQA always uses shared paged KV index layout
    )

    if out is None:
        out_shape = query.shape[:-1] + (kv_lora_rank,)
        out = torch.empty(out_shape, dtype=torch.bfloat16, device=query.device)
    else:
        batch_size, _, num_q_heads, _ = query.shape
        check_shape_dtype_device(
            out,
            [batch_size, num_q_heads, kv_lora_rank],
            torch.bfloat16,
            query.device,
            "out",
        )

    workspace_u8 = workspace_buffer.view(torch.uint8)
    semaphore = workspace_u8[: 8 * 1024 * 1024]  # reserve 8MB for semaphore
    scratch = workspace_u8[8 * 1024 * 1024 :]
    # This can not be replaced by kv_cache.transpose(1, 2) because the stride is not the same
    kv_cache_new = kv_cache.squeeze(1).unsqueeze(2)
    seq_lens_new = seq_lens.unsqueeze(1)

    xqa_mla(
        query,
        kv_cache_new,
        kv_cache_new,
        block_tables,
        seq_lens_new,
        out,
        scratch,
        semaphore,
        block_size,
        q_scale=bmm1_scale,
        kv_scale=bmm2_scale,
        sm_count=sm_count,
        enable_pdl=enable_pdl,
    )

    return out
