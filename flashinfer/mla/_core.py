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

import functools
import math
import os
import weakref
from dataclasses import dataclass, replace
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)

import torch

from flashinfer.autotuner import (
    AutoTuner,
    DynamicTensorSpec,
    TunableRunner,
    TuningConfig,
    cake_auto_candidate_open,
    make_bucket_mapper,
)
from flashinfer.autotuner import autotuner as _autotuner_module
from flashinfer.jit.core import logger

from ..api_logging import (
    _CAKE_AUTO_CANDIDATES_ENV_VAR,
    flashinfer_api,
    flashinfer_experimental_api,
)
from ..trace.templates.attention import (
    trtllm_batch_decode_mla_trace_dispatch,
    trtllm_batch_decode_sparse_mla_dsv4_trace_dispatch,
    xqa_batch_decode_mla_trace,
)
from ..utils import (
    check_shape_dtype_device,
    check_trtllm_gen_sm107_only_feature,
    device_support_pdl,
    get_compute_capability,
    get_device_sm_count,
    _resolve_trtllm_gen_multi_ctas_kv_counter_buffer,
    is_sm12x_supported,
    log2e,
)
from ._utils import (
    MLAHeadDimensions as MLAHeadDimensions,
    deepseek_mla_dimensions as deepseek_mla_dimensions,
    smaller_mla_dimensions as smaller_mla_dimensions,
    compact_query_mla_dimensions as compact_query_mla_dimensions,
    nope_mla_dimensions as nope_mla_dimensions,
    supported_mla_head_dimensions as supported_mla_head_dimensions,
    _check_mla_query_kv_shape,
    _check_mla_dense_page_table_shape,
    _check_mla_sparse_index_shape,
    _round_to_seq_len_bucket,
)


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


def _sparse_mla_reshape(
    tensor: torch.Tensor, shape: Tuple[int, ...], name: str, *, contiguous: bool = False
) -> torch.Tensor:
    if tensor.is_cuda and torch.cuda.is_current_stream_capturing():
        try:
            result = tensor.view(shape)
        except RuntimeError as error:
            raise ValueError(
                f"{name} normalization during CUDA graph capture requires a zero-copy view; "
                "materialize contiguous input buffers eagerly before warmup and capture"
            ) from error
        if contiguous and not result.is_contiguous():
            raise ValueError(
                f"{name} normalization during CUDA graph capture requires a contiguous view; "
                "materialize contiguous input buffers eagerly before warmup and capture"
            )
        return result
    result = tensor.reshape(shape)
    return result.contiguous() if contiguous else result


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
        indices = _sparse_mla_reshape(
            indices, (batch_size * q_len_per_request, sparse_topk), name
        )
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
        lens = _sparse_mla_reshape(lens, (-1,), f"{name}_lens")
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


def _nvfp4_sparse_mla_workspace(
    workspace_buffer: torch.Tensor,
    *,
    num_tokens: int,
    num_heads: int,
    topk: int,
    extra_topk: int,
    use_prefill: bool,
) -> Tuple[
    int,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    torch.Tensor,
]:
    """Compatibility partitioner; prepared execution owns production scratch.

    Legacy token/capacity preconditions run before loading a compiled module.
    CPB=1 preserves the legacy allocated-split shape; successful partitioning
    uses dimensions and dtypes projected by the compiled plan.
    """
    num_splits = (topk + 63) // 64 + (extra_topk + 63) // 64
    workspace_bytes = workspace_buffer.numel() * workspace_buffer.element_size()
    if use_prefill:
        bytes_per_token = num_heads * 4
    else:
        bytes_per_token = num_heads * num_splits * (512 * 2 + 4)
        bytes_per_token += num_heads * 4
    max_chunk_tokens = min(num_tokens, workspace_bytes // bytes_per_token)
    if max_chunk_tokens < 1:
        phase = "prefill" if use_prefill else "decode"
        raise ValueError(
            f"NVFP4 sparse MLA {phase} requires at least {bytes_per_token} "
            f"workspace bytes for one token, got {workspace_bytes}"
        )
    if max_chunk_tokens != num_tokens:
        phase = "prefill" if use_prefill else "decode"
        raise ValueError(
            f"NVFP4 sparse MLA {phase} requires workspace for all query "
            f"tokens: need {bytes_per_token * num_tokens} bytes, got "
            f"{workspace_bytes}"
        )

    from ._sparse_mla_sm120._execution import dsv4_nvfp4_format_info, resolve_dsv4_nvfp4
    from ._sparse_mla_sm120._prepared import _workspace_tensor_view, device_caps

    info = dsv4_nvfp4_format_info()
    page = info["page_size"]
    stride = page * info["bytes_per_token"]
    sm_count, shared = device_caps(workspace_buffer.device)
    plan = resolve_dsv4_nvfp4(
        tokens=num_tokens,
        heads=num_heads,
        topk=topk,
        extra_topk=extra_topk,
        page_size=page,
        extra_page_size=page if extra_topk else 0,
        page_stride_bytes=stride,
        extra_page_stride_bytes=stride if extra_topk else 0,
        cpb=1,
        sm_count=sm_count,
        max_shared_bytes=shared,
        prefill=use_prefill,
    )
    offset = 0
    views: list = []
    for shape, dtype, size, alignment in plan.workspace():
        if not size:
            views.append(None)
            continue
        view, offset = _workspace_tensor_view(
            workspace_buffer,
            byte_offset=offset,
            shape=tuple(shape),
            dtype=getattr(torch, str(dtype)),
            alignment=alignment,
        )
        if view is None:
            raise ValueError("NVFP4 sparse MLA requires workspace for all query tokens")
        views.append(view)
    mid_out, mid_lse, scratch_lse = views
    if mid_out is not None:
        mid_out = mid_out.view(num_tokens, num_heads, -1, info["value_dim"])
    if mid_lse is not None:
        mid_lse = mid_lse.view(num_tokens, num_heads, -1)
    if scratch_lse is None:
        raise ValueError("NVFP4 sparse MLA LSE workspace partition failed")
    return num_tokens, mid_out, mid_lse, scratch_lse


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
    lse_scale: float,
    kv_scale_format: str,
    kv_cache_format: Literal[
        "fp8",
        "nvfp4",
        "fp8_dsv41",
        "fp8_dsv41_fp4_ca",
        "fp8_dsv41_inline",
        "fp8_dsv41_inline_fp4_ca",
    ] = "fp8",
    backend: Literal["sparse", "cake"] = "sparse",
    enable_pdl: Optional[bool] = None,
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

    query_flat = _sparse_mla_reshape(
        query, (batch_size * q_len_per_request, num_heads, head_dim), "query"
    )
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

    if query_flat.shape[0] == 0:
        # Empty batch (e.g. an EP rank that received no tokens under
        # speculative decoding): the runner rejects max_num_tokens == 0, so
        # return the normalized buffers without touching it.
        if return_lse:
            return out, (
                user_lse
                if user_lse is not None
                else torch.empty(
                    flat_lse_shape, dtype=torch.float32, device=query.device
                )
            )
        return out

    if return_lse and out_lse_arg is None:
        out_lse_arg = torch.empty(
            flat_lse_shape, dtype=torch.float32, device=query.device
        )
    if backend == "cake":
        # Two Cake SM120 families answer to backend="cake": the DSv4 NVFP4 decode
        # (kv_cache_format="nvfp4") and the DeepSeek-V4.1 mixed-cache decode
        # (kv_cache_format="fp8_dsv41_fp4_ca", BF16 numerics route).
        cake_functional_run = _cake_sm120_functional_run(kv_cache_format)
        out_lse = cake_functional_run(
            query_flat,
            kv_cache,
            primary_segment.indices,
            out_flat,
            workspace_buffer,
            float(sm_scale),
            lengths=primary_segment.lengths,
            sink=_normalize_optional_mla_sink(sinks, "backend='cake'"),
            extra=extra_segment.kv_cache if extra_segment is not None else None,
            extra_indices=(
                extra_segment.indices if extra_segment is not None else None
            ),
            extra_lengths=(
                extra_segment.lengths if extra_segment is not None else None
            ),
            lse=out_lse_arg,
            lse_scale=lse_scale,
            # Only the mixed-cache family launches its split merge with PDL; the
            # NVFP4 family has no such launch option.
            **(
                {"enable_pdl": enable_pdl}
                if kv_cache_format == "fp8_dsv41_fp4_ca"
                else {}
            ),
        )
        if return_lse:
            return out, user_lse if user_lse is not None else out_lse
        return out

    from ._sparse_mla_sm120._prepared import functional_run

    out_lse = functional_run(
        query_flat,
        kv_cache,
        primary_segment.indices,
        out_flat,
        workspace_buffer,
        float(sm_scale),
        lengths=primary_segment.lengths,
        sink=_normalize_optional_mla_sink(sinks, "backend='sparse'"),
        extra=extra_segment.kv_cache if extra_segment is not None else None,
        extra_indices=extra_segment.indices if extra_segment is not None else None,
        extra_lengths=extra_segment.lengths if extra_segment is not None else None,
        lse=out_lse_arg,
        kv_scale_format=kv_scale_format,
        is_dsv4_nvfp4=(kv_cache_format == "nvfp4"),
        extra_fp4=kv_cache_format in _DSV41_FP4_CA_KV_CACHE_FORMATS,
        lse_scale=lse_scale,
    )

    if return_lse:
        return out, user_lse if user_lse is not None else out_lse
    return out


def _cake_sm120_functional_run(kv_cache_format: str) -> Callable[..., torch.Tensor]:
    """Select the Cake SM120 sparse-MLA public-API route for ``kv_cache_format``."""
    if kv_cache_format == "fp8_dsv41_fp4_ca":
        from ._sparse_mla_sm120.cake_dsv41_mixed import (
            functional_run as cake_dsv41_mixed_functional_run,
        )

        return cake_dsv41_mixed_functional_run
    from ._sparse_mla_sm120._cake_dsv4_nvfp4 import (
        functional_run as cake_dsv4_nvfp4_functional_run,
    )

    return cake_dsv4_nvfp4_functional_run


def _check_sm120_sparse_v32_kv_cache(
    kv_cache: torch.Tensor, *, glm53_nope: bool = False
) -> torch.Tensor:
    if kv_cache.dtype != torch.uint8:
        raise ValueError(
            "SM120 sparse MLA v32/GLM backend expects packed uint8 kv_cache, "
            f"got {kv_cache.dtype}"
        )
    # v32/GLM_NSA rows are exactly 656B. GLM-5.3 NoPE has a 528B payload and
    # the kernels take the row advance as a runtime stride, so padded rows
    # (e.g. a legacy 656B vLLM pool) are accepted as long as the row starts
    # with the 528B payload.
    min_row_bytes = 528 if glm53_nope else 656
    layout_desc = ">=528 (528B payload, padded rows allowed)" if glm53_nope else "656"
    if kv_cache.ndim == 3:
        if kv_cache.size(-1) < min_row_bytes:
            raise ValueError(
                "SM120 sparse MLA v32/GLM expects packed kv_cache last dim "
                f"{layout_desc}, got {tuple(kv_cache.shape)}"
            )
        return kv_cache
    if kv_cache.ndim == 4:
        if kv_cache.size(1) != 1 or kv_cache.size(-1) < min_row_bytes:
            raise ValueError(
                "SM120 sparse MLA v32/GLM expects public HND kv_cache shape "
                f"[num_pages, 1, page_size, {layout_desc}] or 3D shorthand "
                f"[num_pages, page_size, {layout_desc}], got {tuple(kv_cache.shape)}"
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
        return _sparse_mla_reshape(
            seq_lens, (flat_tokens,), "seq_lens", contiguous=True
        )
    if seq_lens.ndim == 1 and seq_lens.numel() == flat_tokens:
        return _sparse_mla_reshape(
            seq_lens, (flat_tokens,), "seq_lens", contiguous=True
        )
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
    lse_scale: float,
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
    rope_v32 = kv_lora_rank == 512 and qk_rope_head_dim == 64 and query.size(-1) == 576
    glm53_nope = (
        kv_lora_rank == 512
        and qk_rope_head_dim == 0
        and query.size(-1) == 512
        and str(kv_scale_format).lower().replace("-", "_") == "arbitrary_fp32"
    )
    if not (rope_v32 or glm53_nope):
        raise ValueError(
            "SM120 sparse MLA expects either the v32/GLM_NSA 512+64 layout "
            "or GLM-5.3 native NoPE 512+0 with arbitrary_fp32 scales; got "
            f"kv_lora_rank={kv_lora_rank}, "
            f"qk_rope_head_dim={qk_rope_head_dim}, query dim={query.size(-1)}, "
            f"kv_scale_format={kv_scale_format!r}"
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

    kv_cache = _check_sm120_sparse_v32_kv_cache(kv_cache, glm53_nope=glm53_nope)
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
        lse_scale=lse_scale,
        kv_scale_format=kv_scale_format,
    )


def _check_mla_functional_shape(
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
    """Validate shared functional shapes and return normalized 4D KV storage.

    TRTLLM-GEN, CuTe DSL and XQA use these query, KV and page-table checks.
    Backend-specific capability restrictions are validated separately.
    """
    kv_cache = _check_mla_query_kv_shape(
        query, kv_cache, kv_lora_rank, qk_rope_head_dim, batch_size, max_q_len
    )
    if sparse_mla_top_k > 0:
        _check_mla_sparse_index_shape(query, page_table, sparse_mla_top_k)
    else:
        num_seqs = query.shape[0] if query.ndim == 4 else batch_size
        _check_mla_dense_page_table_shape(
            page_table,
            num_seqs,
            page_size,
            uses_shared_paged_kv_idx,
            require_aligned_block_table,
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


_DSV4_ROPE_QUANT_HEADS_PER_GROUP = 8
_DSV4_ROPE_QUANT_HEAD_DIM = 512
_DSV4_ROPE_QUANT_SCALE_ALIGNMENT = 4


def _dsv4_rope_quant_group_layout(num_heads: int) -> Tuple[int, int, int]:
    """Return ``(num_groups, heads_per_group, flattened_group_width)``."""
    if num_heads != 128:
        raise ValueError(f"DSv4 RopeQuant requires 128 query heads, got {num_heads}")
    heads_per_group = _DSV4_ROPE_QUANT_HEADS_PER_GROUP
    num_groups = num_heads // heads_per_group
    group_width = heads_per_group * _DSV4_ROPE_QUANT_HEAD_DIM
    return num_groups, heads_per_group, group_width


def _allocate_dsv4_rope_quant_output(
    num_tokens: int, num_heads: int, device: torch.device
) -> torch.Tensor:
    """Allocate the group-major FP8 RopeQuant output view."""
    num_groups, _, group_width = _dsv4_rope_quant_group_layout(num_heads)
    # Physical O is [group, token, flattened-head]; expose [token, group, K].
    return torch.empty(
        (num_groups, num_tokens, group_width),
        dtype=torch.float8_e4m3fn,
        device=device,
    ).transpose(0, 1)


def _allocate_dsv4_rope_quant_output_scale(
    num_tokens: int, num_heads: int, device: torch.device
) -> torch.Tensor:
    """Allocate the padded, group-major packed UE8M0 scale view."""
    num_groups, heads_per_group, _ = _dsv4_rope_quant_group_layout(num_heads)
    scale_buf_m = (
        (num_tokens + _DSV4_ROPE_QUANT_SCALE_ALIGNMENT - 1)
        // _DSV4_ROPE_QUANT_SCALE_ALIGNMENT
        * _DSV4_ROPE_QUANT_SCALE_ALIGNMENT
    )
    # Physical SF is [group, head-in-group, padded-token]. Each INT32 packs the
    # four UE8M0 exponent bytes for one 512-wide head.
    return torch.zeros(
        (num_groups, heads_per_group, scale_buf_m),
        dtype=torch.int32,
        device=device,
    ).permute(2, 0, 1)[:num_tokens]


def _allocate_dsv4_rope_quant_outputs(
    num_tokens: int, num_heads: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Allocate the group-major output layouts consumed by DSv4 DeepGEMM."""
    return (
        _allocate_dsv4_rope_quant_output(num_tokens, num_heads, device),
        _allocate_dsv4_rope_quant_output_scale(num_tokens, num_heads, device),
    )


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


def _check_nvfp4_page_strides(
    kv_cache: torch.Tensor,
    kv_layout: Literal["HND", "NHD"],
    name: str,
) -> None:
    """Validate the 384-byte logical row inside a possibly padded page."""
    page_dim = 1 if kv_cache.ndim == 3 or kv_layout == "NHD" else 2
    page_size = kv_cache.shape[page_dim]
    if kv_cache.stride(-1) != 1 or kv_cache.stride(page_dim) != 384:
        raise ValueError(
            f"NVFP4 {name} entries must be contiguous inside each page; "
            f"got strides {kv_cache.stride()}"
        )
    if kv_cache.stride(0) < page_size * 384:
        raise ValueError(
            f"NVFP4 {name} page stride {kv_cache.stride(0)} is smaller than "
            f"the logical {page_size * 384}-byte page"
        )


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


def _validate_dsv4_sync_checks(device: torch.device) -> bool:
    if os.environ.get("FLASHINFER_VALIDATE_INPUTS", "0") in ("0", ""):
        return False
    if device.type == "cuda":
        with torch.cuda.device(device):
            if torch.cuda.is_current_stream_capturing():
                return False
    return True


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
    metadata_rows_may_be_fewer: bool = False,
    check_topk_lens_range: bool = True,
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
    """Validate the shared DSv4 sparse-MLA inputs.

    ``metadata_rows_may_be_fewer`` (CAKE) accepts ``sparse_indices`` with
    ``1 <= rows <= sum_q`` and sizes ``sparse_topk_lens`` by those rows;
    ``check_topk_lens_range=False`` skips the TRTLLM-GEN ``[128, capacity]``
    length check for backends that clamp lengths in-kernel.
    """
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
    if metadata_rows_may_be_fewer:
        metadata_rows = sparse_indices.shape[0]
        if not 1 <= metadata_rows <= sum_seq_q:
            raise ValueError(
                "Expected 1 <= sparse_indices.shape[0] <= "
                f"{sum_seq_q} query tokens, got {metadata_rows}"
            )
    elif sparse_indices.shape[:-1] != sparse_indices_prefix_shape:
        raise ValueError(
            "Expected sparse_indices.shape[:-1] == "
            f"{sparse_indices_prefix_shape}, got {sparse_indices.shape[:-1]}"
        )
    else:
        metadata_rows = sum_seq_q
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
        metadata_rows,
        "sparse_topk_lens",
        query.device,
        cum_seq_lens_q,
    )
    if (
        check_topk_lens_range
        and _validate_dsv4_sync_checks(query.device)
        and normalized_sparse_lens.numel() > 0
    ):
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


# DeepSeek-V4.1 storage forms. The ``_inline`` forms hold inline-scale records
# (FlashMLA's V4.1 ABI as of its 2026.09.30 release: each token's data bytes
# followed by its own scale bytes); the others keep the footer-scale pages.
_DSV41_INLINE_KV_CACHE_FORMATS = ("fp8_dsv41_inline", "fp8_dsv41_inline_fp4_ca")
_DSV41_FP4_CA_KV_CACHE_FORMATS = ("fp8_dsv41_fp4_ca", "fp8_dsv41_inline_fp4_ca")
_DSV41_KV_CACHE_FORMATS = (
    "fp8_dsv41",
    "fp8_dsv41_fp4_ca",
) + _DSV41_INLINE_KV_CACHE_FORMATS
_DSV4_KV_CACHE_FORMATS = ("fp8", "nvfp4") + _DSV41_KV_CACHE_FORMATS


def _check_dsv4_kv_cache_format_backend(
    kv_cache_format: str,
    backend: Literal["trtllm-gen", "cute-dsl", "sparse", "cake"],
    *,
    sm120_cake: bool = False,
) -> None:
    """Pair a resolved DSv4 backend with the sparse-cache storage format.

    ``fp8_dsv41`` and the inline-scale ``fp8_dsv41_inline`` /
    ``fp8_dsv41_inline_fp4_ca`` are SM120/SM121 ``backend="sparse"`` only;
    ``fp8_dsv41_fp4_ca`` additionally accepts the Cake mixed-cache decode on SM120/SM121
    (``sm120_cake=True``: ``backend="cake"`` resolved on an SM120-family
    device). The 384-byte NVFP4 ABI runs on SM120/SM121 through
    ``backend="sparse"`` or ``backend="cake"`` and on SM100/SM103 through the
    CAKE NVFP4 route (``backend="cake"``); the architecture itself is enforced
    by :func:`_resolve_dsv4_sparse_mla_backend`.
    """
    if kv_cache_format not in _DSV4_KV_CACHE_FORMATS:
        raise ValueError(
            "kv_cache_format must be 'fp8', 'fp8_dsv41', 'fp8_dsv41_fp4_ca', "
            "'fp8_dsv41_inline', 'fp8_dsv41_inline_fp4_ca', or 'nvfp4', got "
            f"{kv_cache_format!r}"
        )
    if kv_cache_format == "fp8_dsv41" and backend != "sparse":
        raise ValueError("kv_cache_format='fp8_dsv41' requires backend='sparse'")
    if kv_cache_format in _DSV41_INLINE_KV_CACHE_FORMATS and backend != "sparse":
        raise ValueError(
            f"kv_cache_format={kv_cache_format!r} (inline-scale records) requires "
            "backend='sparse'; backend='cake' reads the footer-scale "
            "'fp8_dsv41_fp4_ca' cache only"
        )
    if kv_cache_format == "fp8_dsv41_fp4_ca" and backend != "sparse":
        if backend != "cake" or not sm120_cake:
            raise ValueError(
                "kv_cache_format='fp8_dsv41_fp4_ca' requires backend='sparse' (or "
                "backend='cake' on SM120/SM121)"
            )
    if kv_cache_format == "nvfp4" and backend not in ("sparse", "cake"):
        raise ValueError(
            "kv_cache_format='nvfp4' requires backend='sparse' (SM120/SM121) or "
            f"backend='cake' (SM120/SM121 or SM100/SM103), got backend={backend!r}"
        )


def _resolve_dsv4_sparse_mla_backend(
    device: torch.device,
    requested_backend: Literal[
        "auto", "trtllm-gen", "cute-dsl", "sparse", "cake"
    ] = "auto",
) -> Literal["trtllm-gen", "cute-dsl", "sparse", "cake"]:
    cc = get_compute_capability(device)
    is_sm100_family = cc in ((10, 0), (10, 3), (10, 7))
    # The SM100 CAKE kernels build for SM100/SM103 and, for the FP8 decode
    # route, SM107 (run_cake_dsv4 refuses a route SM107 has no program for).
    is_cake_sm100_family = is_sm100_family
    is_sm120_family = cc in ((12, 0), (12, 1))
    if requested_backend == "auto":
        if is_sm120_family:
            return "sparse"
        if is_sm100_family:
            return "trtllm-gen"
        raise ValueError(
            "trtllm_batch_decode_sparse_mla_dsv4 supports SM100/SM103/SM107 via "
            f"TRTLLM-GEN or SM120/SM121 via sparse backend, got SM{cc[0]}{cc[1]}"
        )
    if requested_backend not in ("trtllm-gen", "cute-dsl", "sparse", "cake"):
        raise ValueError(
            "backend must be one of 'auto', 'trtllm-gen', 'cute-dsl', "
            f"'sparse', or 'cake', got {requested_backend!r}"
        )
    if requested_backend == "trtllm-gen" and not is_sm100_family:
        raise ValueError(
            f"backend={requested_backend!r} requires SM100/SM103/SM107, got SM{cc[0]}{cc[1]}"
        )
    if requested_backend == "cute-dsl" and not is_sm100_family:
        raise ValueError(
            f"backend='cute-dsl' requires SM100/SM103/SM107, got SM{cc[0]}{cc[1]}"
        )
    if requested_backend == "sparse" and not is_sm120_family:
        raise ValueError(f"backend='sparse' requires SM120/SM121, got SM{cc[0]}{cc[1]}")
    if requested_backend == "cake" and not (is_cake_sm100_family or is_sm120_family):
        raise ValueError(
            "backend='cake' requires SM100/SM103/SM107 or SM120/SM121, "
            f"got SM{cc[0]}{cc[1]}"
        )
    return cast(Literal["trtllm-gen", "cute-dsl", "sparse", "cake"], requested_backend)


def _is_sm120_family(device: torch.device) -> bool:
    return get_compute_capability(device) in ((12, 0), (12, 1))


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
    lse_scale: float,
    sinks: Optional[torch.Tensor],
    kv_layout: Literal["HND", "NHD"],
    kv_cache_format: Literal[
        "fp8",
        "nvfp4",
        "fp8_dsv41",
        "fp8_dsv41_fp4_ca",
        "fp8_dsv41_inline",
        "fp8_dsv41_inline_fp4_ca",
    ],
    backend: Literal["sparse", "cake"] = "sparse",
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    if bmm2_scale != 1.0:
        raise ValueError("SM120 DSv4 sparse MLA does not support bmm2_scale")
    cake = backend == "cake"
    cake_mixed = cake and kv_cache_format == "fp8_dsv41_fp4_ca"
    if cake and kv_cache_format not in ("nvfp4", "fp8_dsv41_fp4_ca"):
        raise ValueError(
            "backend='cake' on SM120/SM121 serves the DSv4 NVFP4 cache "
            "(kv_cache_format='nvfp4') and the footer-scale DeepSeek-V4.1 mixed "
            f"cache (kv_cache_format='fp8_dsv41_fp4_ca'); got {kv_cache_format!r}"
        )
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
    if cake_mixed:
        from ._sparse_mla_sm120.cake_dsv41_mixed import (
            cake_sparse_mla_sm120_dsv41_mixed_supported_heads,
        )

        cake_heads = cake_sparse_mla_sm120_dsv41_mixed_supported_heads()
        if num_heads not in cake_heads:
            raise ValueError(
                "backend='cake' SM120 DSv4.1 mixed-cache sparse MLA supports "
                f"{cake_heads} query heads, got {num_heads}"
            )
    elif cake:
        from ._sparse_mla_sm120._cake_dsv4_nvfp4 import (
            cake_sparse_mla_sm120_dsv4_nvfp4_supported_heads,
        )

        cake_heads = cake_sparse_mla_sm120_dsv4_nvfp4_supported_heads()
        if num_heads not in cake_heads:
            raise ValueError(
                f"backend='cake' SM120 DSv4 NVFP4 sparse MLA supports {cake_heads} "
                f"query heads, got {num_heads}"
            )
    elif num_heads not in (8, 16, 32, 64, 128):
        raise ValueError(
            "Expected 8, 16, 32, 64, or 128 query heads for SM120 DSv4 "
            f"sparse MLA, got {num_heads}"
        )
    if swa_topk_lens is None:
        raise ValueError(f"backend={backend!r} requires swa_topk_lens")
    if kv_cache_format == "nvfp4" and num_heads == 8 and not cake:
        raise ValueError("NVFP4 sparse MLA does not yet support 8 query heads")

    # Packed FP8 row width per cache format: DSV4 is 584B (448B FP8 + 128B
    # BF16 rope + 8B UE8M0 footer); DSV4.1 is 528B (512B all-FP8 K + 16B
    # 32-wide-group UE8M0 scales, as a page footer or inline after each token's
    # data). The mixed "fp4_ca" forms keep the 528B main (SWA) cache and carry
    # the compressed cache as 288B V41_FP4 rows (256B E2M1 + 32B E4M3-G16
    # scales).
    packed_row_bytes = 528 if kv_cache_format in _DSV41_KV_CACHE_FORMATS else 584
    packed_row_bytes_extra = (
        288 if kv_cache_format in _DSV41_FP4_CA_KV_CACHE_FORMATS else packed_row_bytes
    )

    swa_kv_cache = _check_sm120_dsv4_kv_cache_layout(
        swa_kv_cache, kv_layout, "swa_kv_cache"
    )
    if kv_cache_format == "nvfp4":
        if swa_kv_cache.dtype != torch.uint8 or swa_kv_cache.size(-1) != 384:
            raise ValueError(
                "Expected NVFP4 SM120 DSv4 swa_kv_cache with dtype uint8 and "
                f"head dim 384, got {swa_kv_cache.dtype} "
                f"{tuple(swa_kv_cache.shape)}"
            )
        _check_nvfp4_page_strides(swa_kv_cache, kv_layout, "swa_kv_cache")
        primary_page_size = (
            swa_kv_cache.shape[1]
            if swa_kv_cache.ndim == 3 or kv_layout == "NHD"
            else swa_kv_cache.shape[2]
        )
        if cake:
            # The Cake kernels take the page size and page stride at runtime.
            if primary_page_size < 1 or swa_kv_cache.stride(0) % 16:
                raise ValueError(
                    "backend='cake' NVFP4 primary cache needs a positive page size "
                    "and a 16-byte multiple page stride, got page_size="
                    f"{primary_page_size} stride={swa_kv_cache.stride(0)}"
                )
        elif primary_page_size != 64:
            raise ValueError(
                "NVFP4 sparse MLA primary cache requires page_size=64, got "
                f"{primary_page_size}"
            )
    else:
        if cake_mixed and swa_kv_cache.dtype != torch.uint8:
            raise ValueError(
                "backend='cake' with kv_cache_format='fp8_dsv41_fp4_ca' reads the packed "
                f"uint8 {packed_row_bytes}-byte swa_kv_cache only, got {swa_kv_cache.dtype}"
            )
        if swa_kv_cache.dtype == torch.uint8:
            if swa_kv_cache.size(-1) != packed_row_bytes:
                raise ValueError(
                    f"Expected packed SM120 {kv_cache_format} swa_kv_cache head dim "
                    f"{packed_row_bytes}, got {swa_kv_cache.size(-1)}"
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
            f"together for backend={backend!r}"
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
        if kv_cache_format == "nvfp4":
            if (
                compressed_kv_cache.dtype != torch.uint8
                or compressed_kv_cache.size(-1) != 384
            ):
                raise ValueError(
                    "Expected NVFP4 SM120 DSv4 compressed_kv_cache with dtype "
                    "uint8 and head dim 384, got "
                    f"{compressed_kv_cache.dtype} "
                    f"{tuple(compressed_kv_cache.shape)}"
                )
            _check_nvfp4_page_strides(
                compressed_kv_cache, kv_layout, "compressed_kv_cache"
            )
            extra_page_size = (
                compressed_kv_cache.shape[1]
                if compressed_kv_cache.ndim == 3 or kv_layout == "NHD"
                else compressed_kv_cache.shape[2]
            )
            if cake:
                if extra_page_size < 1 or compressed_kv_cache.stride(0) % 16:
                    raise ValueError(
                        "backend='cake' NVFP4 extra cache needs a positive page "
                        "size and a 16-byte multiple page stride, got page_size="
                        f"{extra_page_size} stride={compressed_kv_cache.stride(0)}"
                    )
            elif extra_page_size not in (2, 64):
                raise ValueError(
                    "NVFP4 sparse MLA extra cache requires page_size 2 or 64, "
                    f"got {extra_page_size}"
                )
        else:
            if cake_mixed and compressed_kv_cache.dtype != torch.uint8:
                raise ValueError(
                    "backend='cake' with kv_cache_format='fp8_dsv41_fp4_ca' reads the packed "
                    f"uint8 {packed_row_bytes_extra}-byte compressed_kv_cache only, got "
                    f"{compressed_kv_cache.dtype}"
                )
            if compressed_kv_cache.dtype == torch.uint8:
                if compressed_kv_cache.size(-1) != packed_row_bytes_extra:
                    raise ValueError(
                        f"Expected packed SM120 {kv_cache_format} compressed_kv_cache "
                        f"head dim {packed_row_bytes_extra}, got "
                        f"{compressed_kv_cache.size(-1)}"
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
            lse_scale=lse_scale,
            # The V4.1 forms select the DSV4_1 model types (32-wide UE8M0
            # groups) in the SM120 runner: DSV4_1 for the footer-scale pages,
            # DSV4_1_INLINE for the inline-scale records; 'fp8' stays on the
            # DSV4 default.
            kv_scale_format=(
                "ue8m0_g32_inline"
                if kv_cache_format in _DSV41_INLINE_KV_CACHE_FORMATS
                else "ue8m0_g32"
                if kv_cache_format in _DSV41_KV_CACHE_FORMATS
                else "auto"
            ),
            kv_cache_format=kv_cache_format,
            backend=backend,
            enable_pdl=enable_pdl,
        ),
    )
    if query.ndim == 3:
        return result.squeeze(1)
    return result


_DSV4_HCA_WINDOW_CAPACITY = 128


@dataclass(frozen=True)
class DSV4HCAMetadata:
    """Reusable metadata for the CuTe DSL causal HCA backend.

    ``hca_swa_indices`` contains absolute rows into the flattened SWA cache;
    ``hca_compressed_block_tables`` contains physical compressed-cache page
    IDs. Both are ordered by flattened query row (``B * Q``).
    ``hca_seq_lens`` describes the backing HCA footprint scheduled by TMA and
    may exceed the effective per-row ``sparse_topk_lens``.
    """

    hca_swa_indices: torch.Tensor
    hca_compressed_block_tables: torch.Tensor
    hca_seq_lens: torch.Tensor
    swa_topk_lens: torch.Tensor
    hca_is_causal: bool = True


def _check_hca_page_size(page_size: int, name: str) -> None:
    if page_size <= 1 or _DSV4_HCA_WINDOW_CAPACITY % page_size != 0:
        raise ValueError(
            f"{name} must be greater than 1 and divide "
            f"{_DSV4_HCA_WINDOW_CAPACITY}, got {page_size}"
        )


def _reject_hca_conversion_during_capture(device: torch.device) -> None:
    if device.type != "cuda":
        return
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "compressed-page-aligned HCA metadata conversion is not CUDA Graph "
                "capture safe; precompute HCA metadata before capture"
            )


_DSV4_HCA_CANONICAL_CHECK_MAX_ELEMENTS = 4 * 1024 * 1024


def _recover_hca_page_ids(
    segment: torch.Tensor,
    active_lens: torch.Tensor,
    *,
    max_active_len: int,
    page_size: int,
    num_pages: int,
    name: str,
) -> torch.Tensor:
    """Recover active page IDs while bounding strict-validation temporaries."""
    rows = segment.shape[0]
    if max_active_len == 0:
        return torch.empty((rows, 0), dtype=torch.int32, device=segment.device)
    if max_active_len > segment.shape[1]:
        raise ValueError(
            f"sparse_indices capacity is too small for the active {name} segment"
        )

    page_offsets = torch.arange(
        0,
        max_active_len,
        page_size,
        dtype=torch.int64,
        device=segment.device,
    )
    page_starts = segment.index_select(1, page_offsets)
    page_ids = torch.div(page_starts, page_size, rounding_mode="floor")
    active_pages = page_offsets.unsqueeze(0) < active_lens.unsqueeze(1)
    invalid = (
        active_pages
        & (
            (torch.remainder(page_starts, page_size) != 0)
            | (page_ids < 0)
            | (page_ids >= num_pages)
        )
    ).any()

    chunk_size = max(
        1,
        min(
            max_active_len,
            _DSV4_HCA_CANONICAL_CHECK_MAX_ELEMENTS // max(rows, 1),
        ),
    )
    for start in range(0, max_active_len, chunk_size):
        stop = min(start + chunk_size, max_active_len)
        positions = torch.arange(start, stop, dtype=torch.int64, device=segment.device)
        values = segment[:, start:stop]
        expected_page_ids = page_ids.index_select(
            1, torch.div(positions, page_size, rounding_mode="floor")
        )
        expected_offsets = torch.remainder(positions, page_size).to(segment.dtype)
        active = positions.unsqueeze(0) < active_lens.unsqueeze(1)
        canonical = (
            torch.div(values, page_size, rounding_mode="floor") == expected_page_ids
        ) & (torch.remainder(values, page_size) == expected_offsets.unsqueeze(0))
        invalid = invalid | (active & ~canonical).any()

    if invalid.item():
        raise ValueError(
            f"active {name} sparse_indices are not a canonical {name}-page "
            "expansion with valid physical page IDs"
        )
    return torch.where(active_pages, page_ids, torch.zeros_like(page_ids)).to(
        torch.int32
    )


def _convert_compressed_page_aligned_sparse_indices_to_hca_metadata(
    sparse_indices: torch.Tensor,
    sparse_topk_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    q_len: int,
    compressed_page_size: int,
    num_swa_rows: int,
    num_compressed_pages: int,
) -> DSV4HCAMetadata:
    """Build HCA gather indices and a compressed page table from token rows."""
    _reject_hca_conversion_during_capture(sparse_indices.device)
    if q_len <= 0:
        raise ValueError(f"q_len must be positive, got {q_len}")
    _check_hca_page_size(compressed_page_size, "compressed page size")
    if num_swa_rows <= 0 or num_compressed_pages <= 0:
        raise ValueError("SWA and compressed KV caches must each contain data")

    for tensor, name in (
        (sparse_indices, "sparse_indices"),
        (sparse_topk_lens, "sparse_topk_lens"),
        (seq_lens, "seq_lens"),
    ):
        if tensor.dtype != torch.int32:
            raise ValueError(f"{name} must have dtype torch.int32, got {tensor.dtype}")
        if tensor.device != sparse_indices.device:
            raise ValueError(
                f"{name} must be on {sparse_indices.device}, got {tensor.device}"
            )
    if sparse_indices.ndim != 2:
        raise ValueError(
            f"sparse_indices must be 2D, got shape {tuple(sparse_indices.shape)}"
        )
    if seq_lens.ndim != 1 or seq_lens.numel() == 0:
        raise ValueError(
            f"seq_lens must be non-empty and 1D, got {tuple(seq_lens.shape)}"
        )
    batch_size = seq_lens.numel()
    rows = batch_size * q_len
    if sparse_indices.shape[0] != rows:
        raise ValueError(
            f"sparse_indices must have B * Q = {rows} rows, got "
            f"{sparse_indices.shape[0]}"
        )
    if sparse_topk_lens.shape != (rows,):
        raise ValueError(
            f"sparse_topk_lens must have shape ({rows},), got "
            f"{tuple(sparse_topk_lens.shape)}"
        )
    sparse_capacity = sparse_indices.shape[1]
    if sparse_capacity < _DSV4_HCA_WINDOW_CAPACITY or sparse_capacity % 4 != 0:
        raise ValueError(
            "sparse_indices capacity must include 128 SWA entries and be a "
            f"multiple of 4, got {sparse_capacity}"
        )

    query_positions = torch.arange(
        q_len, dtype=torch.int32, device=sparse_indices.device
    )
    visible_raw_lens = seq_lens.unsqueeze(1) - q_len + query_positions.unsqueeze(0) + 1
    if (visible_raw_lens <= 0).any().item():
        raise ValueError("seq_lens must be at least q_len for causal dense queries")
    swa_topk_lens = visible_raw_lens.clamp(max=_DSV4_HCA_WINDOW_CAPACITY).reshape(-1)
    expected_sparse_topk_lens = (
        _DSV4_HCA_WINDOW_CAPACITY
        + torch.div(
            visible_raw_lens,
            _DSV4_HCA_WINDOW_CAPACITY,
            rounding_mode="floor",
        )
    ).reshape(-1)
    if (expected_sparse_topk_lens > sparse_capacity).any().item():
        raise ValueError(
            "sparse_indices capacity is too small for the HCA lengths derived "
            "from seq_lens"
        )
    if (sparse_topk_lens != expected_sparse_topk_lens).any().item():
        raise ValueError(
            "hca_sparse_indices_format='compressed-page-aligned' requires "
            "sparse_topk_lens == 128 + floor(visible_raw_len / 128)"
        )
    hca_seq_lens = (
        _DSV4_HCA_WINDOW_CAPACITY
        + torch.div(seq_lens, _DSV4_HCA_WINDOW_CAPACITY, rounding_mode="floor")
    ).contiguous()

    window_indices = sparse_indices[:, :_DSV4_HCA_WINDOW_CAPACITY]
    window_columns = torch.arange(
        _DSV4_HCA_WINDOW_CAPACITY,
        dtype=torch.int32,
        device=sparse_indices.device,
    )
    active_window = window_columns.unsqueeze(0) < swa_topk_lens.unsqueeze(1)
    invalid_window = active_window & (
        (window_indices < 0) | (window_indices >= num_swa_rows)
    )
    if invalid_window.any().item():
        raise ValueError(
            "active SWA sparse_indices must be absolute token rows in "
            f"[0, {num_swa_rows})"
        )
    # Gather4 reads every coordinate before applying window_valid_lens. Replace
    # conventional -1/INT_MAX padding with a legal row for masked positions.
    hca_swa_indices = torch.where(
        active_window, window_indices, torch.zeros_like(window_indices)
    ).contiguous()

    compressed_indices = sparse_indices[:, _DSV4_HCA_WINDOW_CAPACITY:]
    max_compressed_slots = int((hca_seq_lens.max() - _DSV4_HCA_WINDOW_CAPACITY).item())
    compressed_tiles = (
        max_compressed_slots + _DSV4_HCA_WINDOW_CAPACITY - 1
    ) // _DSV4_HCA_WINDOW_CAPACITY
    compressed_pages_per_tile = _DSV4_HCA_WINDOW_CAPACITY // compressed_page_size
    compressed_table_columns = max(1, compressed_tiles * compressed_pages_per_tile)
    hca_compressed_block_tables = torch.zeros(
        (rows, compressed_table_columns),
        dtype=torch.int32,
        device=sparse_indices.device,
    )
    active_compressed_lens = expected_sparse_topk_lens - _DSV4_HCA_WINDOW_CAPACITY
    recovered_compressed_page_ids = _recover_hca_page_ids(
        compressed_indices,
        active_compressed_lens,
        max_active_len=max_compressed_slots,
        page_size=compressed_page_size,
        num_pages=num_compressed_pages,
        name="compressed",
    )
    hca_compressed_block_tables[:, : recovered_compressed_page_ids.shape[1]] = (
        recovered_compressed_page_ids
    )

    return DSV4HCAMetadata(
        hca_swa_indices=hca_swa_indices,
        hca_compressed_block_tables=hca_compressed_block_tables.contiguous(),
        hca_seq_lens=hca_seq_lens,
        swa_topk_lens=swa_topk_lens.contiguous(),
    )


def convert_compressed_page_aligned_sparse_indices_to_hca_metadata(
    sparse_indices: torch.Tensor,
    sparse_topk_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    swa_kv_cache: torch.Tensor,
    compressed_kv_cache: torch.Tensor,
    *,
    q_len: int,
    kv_layout: Literal["HND", "NHD"] = "HND",
) -> DSV4HCAMetadata:
    r"""Convert combined sparse indices into reusable HCA metadata.

    Active SWA entries remain arbitrary absolute token-row indices and may
    express ring-buffer rotation or wraparound. The compressed segment must be
    the canonical expansion ``page_id * page_size + page_offset``. For a dense
    causal query, this conversion requires
    ``visible_raw_len = seq_len - q_len + query_position + 1`` and
    ``sparse_topk_len = 128 + floor(visible_raw_len / 128)``.

    The conversion validates tensor values, allocates metadata, and
    synchronizes the device. It is not CUDA Graph capture safe and must not be
    repeated in a latency-sensitive decode loop. Run it during setup, then
    reuse the returned indices and compressed block table with
    ``backend="cute-dsl"``.
    """
    normalized_swa_cache = _normalize_dsv4_sparse_mla_kv_cache(
        swa_kv_cache, kv_layout, "swa_kv_cache"
    )
    normalized_compressed_cache = _normalize_dsv4_sparse_mla_kv_cache(
        compressed_kv_cache, kv_layout, "compressed_kv_cache"
    )
    for cache, name in (
        (normalized_swa_cache, "swa_kv_cache"),
        (normalized_compressed_cache, "compressed_kv_cache"),
    ):
        if cache.device != sparse_indices.device:
            raise ValueError(
                f"{name} must be on {sparse_indices.device}, got {cache.device}"
            )
    return _convert_compressed_page_aligned_sparse_indices_to_hca_metadata(
        sparse_indices,
        sparse_topk_lens,
        seq_lens,
        q_len=q_len,
        compressed_page_size=normalized_compressed_cache.shape[2],
        num_swa_rows=normalized_swa_cache.shape[0] * normalized_swa_cache.shape[2],
        num_compressed_pages=normalized_compressed_cache.shape[0],
    )


@flashinfer_api(trace=trtllm_batch_decode_sparse_mla_dsv4_trace_dispatch)
def trtllm_batch_decode_sparse_mla_dsv4(
    query: torch.Tensor,
    swa_kv_cache: torch.Tensor,
    workspace_buffer: torch.Tensor,
    sparse_indices: Optional[torch.Tensor] = None,
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
    backend: Literal["auto", "trtllm-gen", "cute-dsl", "sparse", "cake"] = "auto",
    hca_swa_indices: Optional[torch.Tensor] = None,
    hca_compressed_block_tables: Optional[torch.Tensor] = None,
    hca_seq_lens: Optional[torch.Tensor] = None,
    hca_is_causal: bool = True,
    hca_use_persistent: bool = False,
    hca_sparse_indices_format: Optional[Literal["compressed-page-aligned"]] = None,
    remapped_sparse_indices_buffer: Optional[torch.Tensor] = None,
    sparse_indices_are_storage_offsets: Optional[bool] = None,
    dsv4_inv_rope_cos_sin_cache: Optional[torch.Tensor] = None,
    dsv4_output_scale: Optional[torch.Tensor] = None,
    sparse_topk_lens_offset: int = 0,
    multi_ctas_kv_counter_buffer: Optional[torch.Tensor] = None,
    *,
    kv_cache_format: Literal[
        "fp8",
        "nvfp4",
        "fp8_dsv41",
        "fp8_dsv41_fp4_ca",
        "fp8_dsv41_inline",
        "fp8_dsv41_inline_fp4_ca",
    ] = "fp8",
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Decode DeepSeek V4 sparse MLA.

    The implementation is selected from the query device architecture.

    On SM100/SM103/SM107, this calls the prebuilt DeepSeek V4 sparse MLA cubins.
    The query and both KV pools use head dim 512. The query may be BF16 or
    per-tensor FP8 E4M3 and the default output is BF16. When
    ``dsv4_inv_rope_cos_sin_cache`` is provided, the fixed TRTLLM-GEN
    RopeQuant epilogue instead writes group-major FP8 E4M3 values and packed
    UE8M0 scales. The first 128 columns of
    ``sparse_indices`` are SWA entries into ``swa_kv_cache``; remaining columns
    are compressed entries into ``compressed_kv_cache``. ``sparse_topk_lens``
    gives the total active sparse length for each query token and must include
    the fixed 128 SWA entries. ``seq_lens`` provides the original KV sequence
    length for the SWA validity window.

    On SM120/SM121, this calls the packed sparse backend. ``swa_kv_cache`` is
    the required packed uint8 SWA pool: 584 bytes per token for FP8, 528 for
    FP8 DSv4.1 (``kv_cache_format="fp8_dsv41"``), or 384 bytes per token for
    group-16 NVFP4, selected by ``kv_cache_format``.
    ``sparse_indices`` and ``swa_topk_lens`` describe the active SWA segment.
    To add a compressed segment, pass ``compressed_kv_cache`` as another pool
    in the same format and pass ``extra_sparse_indices`` with
    ``extra_sparse_topk_lens``. The SM120/SM121 path accepts BF16 query tensors
    and produces BF16 output. Empty effective KV rows produce zero output;
    the packed backend's final base-2 LSE and finite-sink semantics are
    documented in :meth:`SparseMLASm120Wrapper.run`.

    With ``backend="cute-dsl"`` on SM100/SM103/SM107, this calls the DeepSeek V4
    HCA kernel. Its SWA stream consumes arbitrary physical token-row indices,
    matching the TRTLLM-GEN dynamic-token-sparse ABI, while its compressed
    stream consumes physical page IDs. HCA currently accepts dense FP8 E4M3
    query/KV tensors and produces BF16 output.

    With ``backend="cake"`` on SM120/SM121 and ``kv_cache_format="nvfp4"``,
    this calls the Cake SM120 NVFP4 sparse-MLA kernels through the
    SM120 ``"sparse"`` call surface (``swa_topk_lens``, optional
    ``compressed_kv_cache`` + ``extra_sparse_indices`` /
    ``extra_sparse_topk_lens``, ``sinks``, HND / NHD / 3-D packed caches).
    That route takes the main and extra page sizes at runtime (any positive
    page size, 16-byte multiple page stride) and serves 8, 16, 32, 48, 64, 80,
    96, 112 and 128 query heads; ``workspace_buffer`` holds the split-K
    partials and the LSE (size it with
    :func:`flashinfer.mla.cake_sparse_mla_sm120_dsv4_nvfp4_scratch_bytes`).
    Every call runs either the split decode kernel or the single-launch
    prefill kernel (one CTA per token and head block over all of its
    candidates, no split scratch) as chosen by
    :func:`flashinfer.mla.cake_sparse_mla_sm120_dsv4_nvfp4_select_kernel`
    from the measured crossover of the two sm_120a SKUs. Write the NVFP4
    pools with the fused SM120 writers
    :func:`flashinfer.mla.cake_dsv4_nvfp4_rope_quantize_insert` (sliding-window
    pool: GPT-J RoPE + head-padded ``q_out`` + quantize + insert in one launch)
    and :func:`flashinfer.mla.cake_dsv4_nvfp4_kv_rope_quantize_insert`
    (compressed pool / speculative context), which produce the bytes of
    :func:`flashinfer.mla.nvfp4_quantize_append_sparse_mla_cache` applied to
    the BF16-rounded roped rows.

    With ``backend="cake"`` on SM120/SM121 and
    ``kv_cache_format="fp8_dsv41_fp4_ca"``, this calls the Cake SM120
    DeepSeek-V4.1 mixed-cache sparse-MLA decode kernels: the 528-byte FP8
    main (SWA) pool (``swa_kv_cache``) and the optional 288-byte V41_FP4
    compressed pool (``compressed_kv_cache``), each with its own positive
    runtime page size and 16-byte-multiple page stride, read with the BF16
    numerics route (BF16 query, exact on-chip dequantization of both caches,
    fp32 accumulation and softmax). The call surface is the SM120 ``"sparse"``
    one; ``workspace_buffer`` holds the split-K partials and the LSE (size it
    with :func:`flashinfer.mla.cake_sparse_mla_sm120_dsv41_mixed_scratch_bytes`).
    Write the main pool with
    :func:`flashinfer.mla.dsv41_fp8_quantize_pack_sparse_mla_cache` and the
    compressed pool with
    :func:`flashinfer.mla.dsv41_fp4_quantize_pack_sparse_mla_cache`.

    With ``backend="cake"`` on SM100/SM103, this calls the source-level CAKE
    kernels (``flashinfer.mla.cake_dsv4``); on SM107 the FP8 cache's decode
    route runs them (the FP8 persistent, M64 and low-head prefill programs),
    and any other route raises. The metadata may describe fewer
    tokens than ``query`` provides: with dense ``[B, Q, H, D]`` input the
    table rows cover the first ``T <= B * Q`` flattened tokens (padded batch),
    with ragged ``[sum_q, H, D]`` input the first ``T <= sum_q`` rows. Query
    rows ``>= T`` are neither read nor written (``out`` keeps its prior
    contents there). Metadata may be one combined table (``sparse_indices
    [T, sparse_topk]`` + ``sparse_topk_lens`` counting the 128 SWA slots) or
    two separate tables (``sparse_indices [T, 128]`` as the SWA table +
    ``extra_sparse_indices [T, topk_c]``; lengths either as ``sparse_topk_lens``
    in the combined convention or as ``extra_sparse_topk_lens`` counting
    compressed slots only, which implies a length offset of 128). The kernels
    use ``clamp(len + sparse_topk_lens_offset, 0, sparse_topk)`` entries of
    each row. No copies are made: tables must be int32 with unit column stride,
    start at a 16-byte aligned address and, with more than one row, have a row
    stride that is a multiple of 4 (the kernels load index rows 16 bytes at a
    time), and ``query``/``out``/KV pools must be densely
    packed. ``workspace_buffer`` is carved deterministically and exactly per
    route (:func:`flashinfer.mla.cake_dsv4_workspace_requirement`); any
    buffer from the one-row minimum up works: when one launch over all
    metadata rows does not fit, the rows are tiled into consecutive launches
    (a fixed 128 MiB buffer admits every row count on every route).
    :func:`flashinfer.mla.get_cake_dsv4_workspace_bytes` is the single-launch
    upper bound. The split-merge counters are zeroed as part of the launch
    (first eager use through a workspace, or a zero fill recorded into the
    graph when a capture reaches an unprimed workspace); no host priming
    step is required and none raises inside capture. ``seq_lens`` bounds the
    128 SWA columns of every row exactly as TRTLLM-GEN does: with ``b`` the
    request of a row, ``q_len_b`` its query length and ``q_off`` the row's
    position in it, only the first ``clamp(seq_lens[b] - (q_len_b - 1 -
    q_off), 0, 128)`` SWA columns are attended; ``-1`` slots and columns at
    or beyond the active length are never attended. The only allocation on
    the call path is the output when ``out`` is omitted; with ``out``
    provided, captured graphs replay with unchanged
    ``torch.cuda.memory_allocated()``. Warm up eagerly before capture (JIT
    build, descriptor storage).

    Parameters
    ----------
    query : torch.Tensor
        Dense query input ``[batch_size, q_len_per_request, num_heads, 512]``
        or varlen query input ``[sum_q, num_heads, 512]`` when
        ``cum_seq_lens_q`` is provided. SM100/SM103/SM107 accept BF16 or FP8
        E4M3; SM120/SM121 accepts BF16. With ``backend="cake"`` and
        ``kv_cache_format="nvfp4"`` on SM100/SM103 the query is either BF16
        (the NVFP4 decode family) or packed
        NVFP4, uint8 with last dimension 384 (the cache's row format: codes,
        then the 28 E4M3 scale factors), which runs the NVFP4 prefill bodies
        at any row count; the call packs nothing.
        :func:`flashinfer.mla.cake_dsv4_nvfp4_pack_query`
        packs a BF16 query. The output is BF16 with last dimension 512.
    swa_kv_cache : torch.Tensor
        SWA KV cache. TRTLLM-GEN uses head dim 512; SM120 sparse uses an opaque
        packed uint8 record with last dimension 584 (FP8), 528 (DSV4.1 FP8) or
        384 (NVFP4). The ``fp8_dsv41_fp4_ca`` format additionally carries a
        288-byte compressed V41_FP4 record. Layout follows ``kv_layout``.
    workspace_buffer : torch.Tensor
        Byte workspace used by TRTLLM-GEN or HCA split-K reduction. The
        TRTLLM-GEN multi-CTA KV counters are managed in a separate internal
        buffer. SM120 partitions this buffer into the resolved plan's partial
        output, split LSE and final LSE. Eager FP8 may allocate temporary scratch
        when capacity is insufficient; NVFP4 and CUDA graph capture reject
        insufficient capacity. Warm the metadata shape eagerly before capture.
        This scratch rule is separate from the documented output allocation
        when ``out`` is omitted; provide ``out`` to avoid that allocation.
        A changed tuning profile requires eager warmup and recapture.
        ``backend="cake"`` carves this buffer as ``[reserved slab |
        split-merge counters | partial_O (split routes) | partial_lse]`` and
        tiles the metadata rows over several launches when one launch does
        not fit; see :func:`flashinfer.mla.get_cake_dsv4_workspace_bytes` for
        the single-launch bound,
        :func:`flashinfer.mla.cake_dsv4_workspace_requirement` for the exact
        per-route numbers and
        :func:`flashinfer.mla.cake_dsv4_workspace_reset` for the optional
        counter reset. CAKE never allocates scratch.
    sparse_indices : Optional[torch.Tensor]
        TRTLLM-GEN combined sparse table, or the SM120 sparse SWA segment.
        For ``backend="cake"`` the combined table ``[T, sparse_topk]``, or the
        ``[T, 128]`` SWA table when ``extra_sparse_indices`` is given; ``T``
        may be smaller than the number of query tokens.
        Pass ``None`` for the explicit ``backend="cute-dsl"`` metadata path.
        Combined HCA tables whose compressed segment is a canonical page
        expansion may instead be converted by setting
        ``hca_sparse_indices_format="compressed-page-aligned"``. SWA entries
        remain arbitrary absolute token rows in this mode.
    compressed_kv_cache : Optional[torch.Tensor]
        Primary/compressed KV cache in the same backend layout as
        ``swa_kv_cache``. Required by ``trtllm-gen`` and HCA, and by SM120
        ``sparse`` when ``extra_sparse_indices`` is provided.
    sparse_topk_lens : Optional[torch.Tensor]
        Flattened total sparse MLA top-k lengths in query-token order, shape
        ``[sum_q]``. Values must already include the fixed 128 SWA entries,
        matching TRTLLM-GEN ``sparseMlaTopkLengths``. For TRTLLM-GEN they must
        not exceed ``sparse_indices.shape[-1]``. HCA also requires this tensor;
        there it describes the visible window-plus-compressed slot count.
        For ``backend="cake"`` it has one entry per metadata row and may be
        replaced by ``extra_sparse_topk_lens``.
    seq_lens : Optional[torch.Tensor]
        Original KV sequence lengths, shape ``[batch_size]`` INT32. Required
        by ``trtllm-gen`` and by compressed-page-aligned HCA metadata
        conversion. ``backend="cake"`` derives the per-token SWA validity
        window from it (see above) and binds it, with ``cum_seq_lens_q`` /
        ``max_q_len`` / the dense batch layout, to every kernel variant.
    out : Optional[torch.Tensor]
        Optional preallocated output. The default path expects the same shape
        as ``query`` and BF16 dtype. RopeQuant expects FP8 E4M3 shape
        ``[sum_q, 16, 4096]`` with strides
        ``(4096, sum_q * 4096, 1)``. If omitted, FlashInfer allocates the
        appropriate layout.
    bmm1_scale : Union[float, torch.Tensor]
        Fused per-tensor scale for QK and softmax. Tensor form must be FP32.
        HCA currently accepts only a Python float.
    bmm2_scale : Union[float, torch.Tensor]
        Fused per-tensor scale for VO. Tensor form must be FP32.
        HCA currently accepts only a Python float.
    sinks : Optional[torch.Tensor]
        Optional attention sink logits, shape ``[num_heads]`` FP32.
    kv_layout : Literal["HND", "NHD"]
        Layout of both KV pools.
    cum_seq_lens_q : Optional[torch.Tensor]
        Cumulative query lengths for varlen query input, shape ``[batch_size + 1]``
        INT32. When provided, dynamic top-k lengths are consumed in flattened
        query-token order. On SM120/SM121 (``sparse`` backend) the query is
        processed per token and per-token sparse indices fully determine
        visibility, so ``cum_seq_lens_q`` and ``max_q_len`` are accepted for
        call-site compatibility but do not affect the result.
    max_q_len : Optional[int]
        Maximum query length in the varlen batch. Required with
        ``cum_seq_lens_q``.
    enable_pdl : Optional[bool]
        Whether to enable Programmatic Dependent Launch. Used by the
        TRTLLM-GEN path; on SM100/SM103/SM107 by ``backend="cake"`` for its
        FP8 persistent, M64 and low-head prefill programs and, on SM100/SM103,
        its FP8 H64 cluster program (the other BF16 / FP8 programs launch
        without it); and on SM120/SM121 by ``backend="cake"`` with
        ``kv_cache_format="fp8_dsv41_fp4_ca"`` (its split merge is launched as a
        programmatic dependent of the decode grid). ``None`` follows the device
        default. The NVFP4 Cake family rejects ``True``.
    swa_topk_lens : Optional[torch.Tensor]
        Active SWA segment lengths, shape ``[sum_q]`` INT32. On SM120/SM121
        these are sparse-segment lengths. HCA requires them as its per-row
        visible sliding-window lengths in the range 0 through 128.
    extra_sparse_indices : Optional[torch.Tensor]
        Optional SM120/SM121 compressed segment indices into
        ``compressed_kv_cache``. With ``backend="cake"``, the separate
        compressed table ``[T, topk_c]`` INT32 (unit column stride, 16-byte
        aligned start and, with more than one row, a row stride that is a
        multiple of 4) that pairs with a ``[T, 128]`` SWA ``sparse_indices``.
    extra_sparse_topk_lens : Optional[torch.Tensor]
        Active compressed segment lengths for SM120/SM121, shape ``[sum_q]``
        INT32. With ``backend="cake"``, compressed-only lengths ``[T]`` that
        replace ``sparse_topk_lens`` (the host adds the 128 SWA slots through
        ``sparse_topk_lens_offset``).
    backend : {"auto", "trtllm-gen", "cute-dsl", "sparse", "cake"}
        Backend selection. ``"auto"`` preserves the architecture-based default:
        the prebuilt cubins on SM100/SM103/SM107 and sparse on SM120/SM121. HCA is selected
        only when ``"cute-dsl"`` is requested explicitly. Source-level CAKE
        kernels are selected only when ``"cake"`` is requested explicitly: on
        SM100/SM103 the FP8/BF16 DSv4 family, on SM120/SM121 the NVFP4 DSv4
        sparse-MLA decode (``kv_cache_format="nvfp4"`` only).
    hca_swa_indices : Optional[torch.Tensor]
        Absolute SWA token-row indices, shape ``[B * Q, 128]`` INT32. Ring
        rotation and wraparound are supported. Every entry, including masked
        padding, must name a legal row of the flattened SWA cache.
    hca_compressed_block_tables : Optional[torch.Tensor]
        HCA compressed-pool page IDs, shape ``[B * Q, max_pages]`` INT32.
        For each query row, the valid page-ID prefix must cover the compressed
        footprint implied by ``hca_seq_lens``, rounded up to 128-slot tiles.
        Those page IDs must be legal even when ``sparse_topk_lens`` masks the
        corresponding slots.
    hca_seq_lens : Optional[torch.Tensor]
        Per-request backing HCA slot counts ``[B]`` INT32 used to schedule TMA
        loads. Each value counts the 128 window slots plus compressed slots,
        not original raw KV tokens, and may exceed an effective per-row
        ``sparse_topk_lens``.
    hca_is_causal : bool
        Must currently be ``True``. HCA index/page-table rows and valid-length
        tensors are per query token (``B * Q`` rows).
    hca_use_persistent : bool
        Select the CuTe DSL persistent tile scheduler. This removes the
        non-persistent ``B * Q <= 65535`` launch-grid restriction.
    hca_sparse_indices_format : Optional[Literal["compressed-page-aligned"]]
        Opt-in compatibility mode for legacy TRTLLM-GEN metadata. With
        ``"compressed-page-aligned"``, active SWA entries remain arbitrary
        absolute token rows, while the active compressed segment must be the
        canonical expansion ``page_id * page_size + page_offset``. The
        dispatcher validates and converts them into HCA gather indices, a
        compressed block table, ``hca_seq_lens``, and ``swa_topk_lens``. This
        tagged path is a one-shot compatibility path: it allocates,
        synchronizes the device, immediately launches the decode, and is not
        CUDA Graph capture safe. Performance-sensitive callers must precompute
        with
        :func:`convert_compressed_page_aligned_sparse_indices_to_hca_metadata`
        and reuse the returned metadata through the explicit HCA arguments.
    remapped_sparse_indices_buffer : Optional[torch.Tensor]
        Optional INT32 output buffer matching ``sparse_indices``. TRTLLM-GEN
        uses it when either KV pool has a strided page layout. Passing a stable
        buffer avoids an allocation in serving and CUDA graph paths. Concurrent
        calls must use distinct buffers. The buffer must not share storage
        with ``sparse_indices`` when remapping is required.
    sparse_indices_are_storage_offsets : Optional[bool]
        Encoding of ``sparse_indices`` for strided TRTLLM-GEN KV pools. Set to
        ``False`` for logical flattened token indices or ``True`` for indices
        already adjusted to storage-row offsets. Strided pools require an
        explicit value to prevent accidental double remapping.
    dsv4_inv_rope_cos_sin_cache : Optional[torch.Tensor]
        Enables the TRTLLM-GEN DSv4 RopeQuant epilogue. Must be a contiguous
        FP32 tensor shaped ``[max_position, 64]`` with each row laid out as
        ``[cos(32), sin(32)]`` for interleaved RoPE. The kernel derives each
        query position as ``seq_len - q_len + local_query_index``; the cache
        must cover every derived position. This mode requires FP8 E4M3 Q/K/V,
        128 query heads, and
        ``backend="trtllm-gen"``.
    dsv4_output_scale : Optional[torch.Tensor]
        Optional preallocated packed UE8M0 scale output. Shape must be
        ``[sum_q, 16, 8]`` with strides
        ``(1, 8 * scale_buf_m, scale_buf_m)``; ``scale_buf_m`` must be a
        multiple of four and at least ``sum_q``. If omitted in RopeQuant mode,
        FlashInfer allocates it with zeroed physical padding. A caller-provided
        buffer must also have its padded token rows zero-initialized before
        first use; the cubin intentionally leaves those rows unwritten.
        Supplying this tensor together with ``out`` avoids RopeQuant output
        allocations. CUDA Graph use still requires normal JIT warmup and stable
        caller-controlled temporary buffers; the existing DSV4 path allocates
        its internal counter buffer per invocation.
    sparse_topk_lens_offset : int
        ``backend="cake"`` only. Added in-kernel to every length entry before
        clamping to ``[0, sparse_topk]``; lets callers pass lengths that do not
        yet include the 128 SWA slots (``128``) or shrink every row uniformly.
        Combined with ``extra_sparse_topk_lens`` the implied 128 is added on
        top of this value.
    multi_ctas_kv_counter_buffer : Optional[torch.Tensor]
        Caller-owned, zero-initialized ``uint8`` buffer for the TRTLLM-GEN
        multi-CTA KV split semaphores, sized by
        :func:`flashinfer.utils.get_trtllm_gen_multi_ctas_kv_counter_bytes`
        (``batch_size``, ``num_heads``, SM count). The kernel resets the
        counters at the end of every launch, so one buffer can be reused across
        launches and CUDA-graph replays. When ``None`` (default) a fresh buffer
        is allocated per call, which is not CUDA-graph friendly. Only used by
        ``backend="trtllm-gen"``.
    kv_cache_format : {"fp8", "nvfp4", "fp8_dsv41", "fp8_dsv41_fp4_ca", "fp8_dsv41_inline", "fp8_dsv41_inline_fp4_ca"}
        SM120/SM121 sparse-cache storage format. ``"fp8"`` preserves the
        existing 584-byte DSv4 cache ABI. ``"fp8_dsv41"`` selects the
        528-byte DeepSeek-V4.1 ABI (all-FP8 512-wide K with a 16-byte
        32-wide-group UE8M0 footer, no BF16 rope segment). ``"nvfp4"`` selects
        the DSV4 384-byte group-16 NVFP4 cache ABI and its prefill/decode
        kernels. NVFP4 currently supports 16/32/64/128 heads, primary top-k
        128 or 512, primary page size 64, and optional extra-cache page size 2
        or 64. ``"fp8_dsv41_fp4_ca"`` is the mixed DeepSeek-V4.1 form: the main
        (SWA) cache keeps the 528-byte FP8 ABI while the compressed cache holds
        288-byte V41_FP4 rows (256B E2M1 + 32B E4M3-G16 scales), upconverted to
        FP8 at gather time by the SM120 kernels. Both V4.1 forms support
        dual-cache prefill for 8/16/32/64 heads. V4.1 uses independent positive
        runtime page sizes for main and extra caches, including 256/64 and
        61/53 layouts; page strides may include padding.
        FP4 upconversion is lossy relative to direct cache dequantization.
        With ``backend="cake"`` on SM120/SM121 the ``fp8_dsv41_fp4_ca`` form
        selects the Cake mixed-cache decode instead (BF16 numerics route,
        both caches dequantized exactly on chip).
        ``"fp8_dsv41_inline"`` and ``"fp8_dsv41_inline_fp4_ca"`` are the same
        two V4.1 forms on inline-scale records, FlashMLA's V4.1 ABI as of its
        2026.09.30 release: each token's 512 data bytes are followed by its own
        16 scale bytes (V41_FP4: 256 data bytes then 32 scale bytes) instead
        of a per-page scale footer. Inline records are rows of the whole cache
        at a uniform row pitch (a 16-byte multiple of at least the record
        size) with a page stride of whole rows; ``sparse_indices`` and
        ``extra_sparse_indices`` then address row ``page * (page_stride //
        row_pitch) + slot`` (the paged slot ID when pages carry no padding).
        They take the same SM120 routes as the footer forms, reading each
        record with one bulk copy (``backend="sparse"`` only; ``backend=
        "cake"`` rejects them). Footer and inline caches cannot be told apart
        from their shapes, so the format must name the layout the cache was
        written in.

    Returns
    -------
    torch.Tensor or tuple[torch.Tensor, torch.Tensor]
        The existing BF16 output when RopeQuant is disabled. With RopeQuant,
        returns ``(out_fp8, out_scale)``. ``out_fp8`` has shape
        ``[sum_q, 16, 4096]`` and group-major strides
        ``(4096, sum_q * 4096, 1)``; ``out_scale`` uses the packed UE8M0
        layout described above.
    """
    backend = _resolve_dsv4_sparse_mla_backend(query.device, backend)
    _check_dsv4_kv_cache_format_backend(
        kv_cache_format, backend, sm120_cake=_is_sm120_family(query.device)
    )

    rope_quant = dsv4_inv_rope_cos_sin_cache is not None
    if dsv4_output_scale is not None and not rope_quant:
        raise ValueError("dsv4_output_scale requires dsv4_inv_rope_cos_sin_cache")
    if rope_quant and backend != "trtllm-gen":
        raise ValueError("DSv4 RopeQuant requires backend='trtllm-gen'")

    if backend != "trtllm-gen" and (
        remapped_sparse_indices_buffer is not None
        or sparse_indices_are_storage_offsets is not None
    ):
        raise ValueError(
            f"backend={backend!r} does not accept remapped_sparse_indices_buffer "
            "or sparse_indices_are_storage_offsets"
        )

    if backend == "cute-dsl":
        if not hca_is_causal:
            raise ValueError("backend='cute-dsl' currently supports causal HCA only")
        if cum_seq_lens_q is not None or max_q_len is not None:
            raise ValueError(
                "backend='cute-dsl' currently supports dense [B, Q, H, D] "
                "queries only; cum_seq_lens_q and max_q_len must be None"
            )
        if extra_sparse_indices is not None or extra_sparse_topk_lens is not None:
            raise ValueError(
                "backend='cute-dsl' uses HCA metadata and does not accept "
                "extra_sparse_indices or extra_sparse_topk_lens"
            )
        if enable_pdl:
            raise ValueError("backend='cute-dsl' HCA does not support enable_pdl")
        if hca_sparse_indices_format not in (None, "compressed-page-aligned"):
            raise ValueError(
                "hca_sparse_indices_format must be None or "
                "'compressed-page-aligned', got "
                f"{hca_sparse_indices_format!r}"
            )

        normalized_swa_cache = _normalize_dsv4_sparse_mla_kv_cache(
            swa_kv_cache, kv_layout, "swa_kv_cache"
        ).squeeze(1)
        flattened_swa_cache = normalized_swa_cache.reshape(
            -1, normalized_swa_cache.shape[-1]
        )
        if compressed_kv_cache is None:
            raise ValueError("backend='cute-dsl' requires compressed_kv_cache")
        normalized_compressed_cache = _normalize_dsv4_sparse_mla_kv_cache(
            compressed_kv_cache, kv_layout, "compressed_kv_cache"
        ).squeeze(1)

        if hca_sparse_indices_format == "compressed-page-aligned":
            generated_inputs = {
                "hca_swa_indices": hca_swa_indices,
                "hca_compressed_block_tables": hca_compressed_block_tables,
                "hca_seq_lens": hca_seq_lens,
                "swa_topk_lens": swa_topk_lens,
            }
            conflicts = [
                name for name, value in generated_inputs.items() if value is not None
            ]
            if conflicts:
                raise ValueError(
                    "hca_sparse_indices_format='compressed-page-aligned' generates "
                    + ", ".join(conflicts)
                    + "; do not pass them explicitly"
                )
            if sparse_indices is None or sparse_topk_lens is None or seq_lens is None:
                raise ValueError(
                    "hca_sparse_indices_format='compressed-page-aligned' requires "
                    "sparse_indices, sparse_topk_lens, and seq_lens"
                )
            if query.ndim != 4:
                raise ValueError(
                    "hca_sparse_indices_format='compressed-page-aligned' requires "
                    "dense query shape [B, Q, H, D]"
                )
            if seq_lens.numel() != query.shape[0]:
                raise ValueError(
                    "hca_sparse_indices_format='compressed-page-aligned' requires "
                    f"seq_lens with {query.shape[0]} entries, got {seq_lens.numel()}"
                )
            for tensor, name in (
                (sparse_indices, "sparse_indices"),
                (sparse_topk_lens, "sparse_topk_lens"),
                (seq_lens, "seq_lens"),
                (normalized_swa_cache, "swa_kv_cache"),
                (normalized_compressed_cache, "compressed_kv_cache"),
            ):
                if tensor.device != query.device:
                    raise ValueError(
                        f"{name} must be on {query.device}, got {tensor.device}"
                    )
            _reject_hca_conversion_during_capture(query.device)
            metadata = _convert_compressed_page_aligned_sparse_indices_to_hca_metadata(
                sparse_indices,
                sparse_topk_lens,
                seq_lens,
                q_len=query.shape[1],
                compressed_page_size=normalized_compressed_cache.shape[1],
                num_swa_rows=flattened_swa_cache.shape[0],
                num_compressed_pages=normalized_compressed_cache.shape[0],
            )
            hca_swa_indices = metadata.hca_swa_indices
            hca_compressed_block_tables = metadata.hca_compressed_block_tables
            hca_seq_lens = metadata.hca_seq_lens
            swa_topk_lens = metadata.swa_topk_lens
        else:
            if sparse_indices is not None:
                raise ValueError(
                    "backend='cute-dsl' consumes explicit HCA metadata, so "
                    "sparse_indices must be None unless "
                    "hca_sparse_indices_format='compressed-page-aligned'"
                )
            if seq_lens is not None:
                raise ValueError(
                    "backend='cute-dsl' uses hca_seq_lens, so seq_lens must be "
                    "None unless "
                    "hca_sparse_indices_format='compressed-page-aligned'"
                )

        required_hca_inputs = {
            "compressed_kv_cache": compressed_kv_cache,
            "sparse_topk_lens": sparse_topk_lens,
            "swa_topk_lens": swa_topk_lens,
            "hca_swa_indices": hca_swa_indices,
            "hca_compressed_block_tables": hca_compressed_block_tables,
            "hca_seq_lens": hca_seq_lens,
        }
        missing = [name for name, value in required_hca_inputs.items() if value is None]
        if missing:
            raise ValueError("backend='cute-dsl' requires " + ", ".join(missing))
        if isinstance(bmm1_scale, torch.Tensor) or isinstance(bmm2_scale, torch.Tensor):
            raise TypeError(
                "backend='cute-dsl' HCA currently requires Python float "
                "bmm1_scale and bmm2_scale"
            )

        from ..cute_dsl.attention.wrappers.batch_hca import cute_dsl_hca_decode

        return cute_dsl_hca_decode(
            query=query,
            window_kv_cache=flattened_swa_cache,
            compressed_kv_cache=normalized_compressed_cache,
            workspace_buffer=workspace_buffer,
            window_indices=hca_swa_indices,
            compressed_block_tables=hca_compressed_block_tables,
            hca_seq_lens=hca_seq_lens,
            sparse_topk_lens=sparse_topk_lens,
            window_valid_lens=swa_topk_lens,
            softmax_scale=float(bmm1_scale),
            output_scale=float(bmm2_scale),
            sinks=sinks,
            out=out,
            is_causal=hca_is_causal,
            is_persistent=hca_use_persistent,
        )

    unexpected_hca_inputs = {
        "hca_swa_indices": hca_swa_indices,
        "hca_compressed_block_tables": hca_compressed_block_tables,
        "hca_seq_lens": hca_seq_lens,
        "hca_is_causal": hca_is_causal if not hca_is_causal else None,
        "hca_use_persistent": hca_use_persistent if hca_use_persistent else None,
        "hca_sparse_indices_format": hca_sparse_indices_format,
    }
    unexpected_hca_input_names = [
        name for name, value in unexpected_hca_inputs.items() if value is not None
    ]
    if unexpected_hca_input_names:
        raise ValueError(
            f"backend={backend!r} does not accept "
            + ", ".join(unexpected_hca_input_names)
        )

    # On SM120/SM121, backend="cake" selects the Cake NVFP4 sparse-MLA decode,
    # which shares the SM120 "sparse" call surface (swa_topk_lens, dual cache).
    cake_sm120 = backend == "cake" and _is_sm120_family(query.device)
    if backend == "cake" and kv_cache_format == "nvfp4":
        # The NVFP4 Cake family has no PDL launch option.  The DeepSeek-V4.1
        # mixed-cache family launches its split merge as a programmatic
        # dependent of the decode grid, and the SM100/SM103 BF16 / FP8 programs
        # that support it launch as programmatic dependents (None = device
        # default for both).
        if enable_pdl:
            raise ValueError(
                "backend='cake' with kv_cache_format='nvfp4' does not support enable_pdl"
            )
        enable_pdl = False
    elif enable_pdl is None:
        enable_pdl = device_support_pdl(query.device)
    if isinstance(bmm1_scale, torch.Tensor):
        if backend == "sparse" or cake_sm120:
            raise ValueError(
                "SM120/SM121 DSv4 sparse MLA expects bmm1_scale to be a float"
            )
        if bmm1_scale.dtype != torch.float32:
            raise TypeError("bmm1_scale tensor must have dtype torch.float32")
        if backend == "trtllm-gen":
            bmm1_scale = bmm1_scale * log2e
    if isinstance(bmm2_scale, torch.Tensor):
        if backend == "sparse" or cake_sm120:
            raise ValueError(
                "SM120/SM121 DSv4 sparse MLA expects bmm2_scale to be a float"
            )
        if bmm2_scale.dtype != torch.float32:
            raise TypeError("bmm2_scale tensor must have dtype torch.float32")

    if backend == "sparse" or cake_sm120:
        if sparse_indices is None:
            raise ValueError(f"backend={backend!r} requires sparse_indices")
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
            lse_scale=1.0,
            sinks=sinks,
            kv_layout=kv_layout,
            kv_cache_format=kv_cache_format,
            backend="cake" if cake_sm120 else "sparse",
            enable_pdl=enable_pdl if cake_sm120 else None,
        )

    if backend == "cake" and kv_cache_format == "nvfp4":
        # SM100/SM103 NVFP4 cache route. Two independent tables
        # (``sparse_indices`` over ``swa_kv_cache``, ``extra_sparse_indices``
        # over ``compressed_kv_cache``) as in the SM120 NVFP4 ABI; the main
        # active length is ``sparse_topk_lens`` (``swa_topk_lens`` is accepted
        # as the SM120 spelling of the same vector). No combined-table offset.
        from .cake_dsv4 import run_cake_dsv4_nvfp4

        if sparse_indices is None:
            raise ValueError("backend='cake' requires sparse_indices")
        if sparse_topk_lens is not None and swa_topk_lens is not None:
            raise ValueError(
                "kv_cache_format='nvfp4' takes either sparse_topk_lens or "
                "swa_topk_lens for the main table, not both"
            )
        if sparse_topk_lens_offset != 0:
            raise ValueError(
                "kv_cache_format='nvfp4' uses independent tables; "
                "sparse_topk_lens_offset must be 0"
            )
        if multi_ctas_kv_counter_buffer is not None:
            raise ValueError(
                "multi_ctas_kv_counter_buffer is only used by backend='trtllm-gen'"
            )
        if query.ndim == 4:
            if max_q_len is not None and max_q_len != query.shape[1]:
                raise ValueError(
                    f"Expected max_q_len == {query.shape[1]} for dense query input, "
                    f"got {max_q_len}"
                )
            query_flat = query.flatten(0, 1)
        elif query.ndim == 3:
            query_flat = query
        else:
            raise ValueError(f"Expected query.ndim == 3 or 4, got {query.ndim}")
        if out is None:
            # A packed NVFP4 query has 384-byte rows; the output rows are 512 BF16.
            out = torch.empty(
                (*query.shape[:-1], 512), dtype=torch.bfloat16, device=query.device
            )
        return run_cake_dsv4_nvfp4(
            query=query_flat,
            swa_kv_cache=swa_kv_cache,
            compressed_kv_cache=compressed_kv_cache,
            workspace_buffer=workspace_buffer,
            sparse_indices=sparse_indices,
            sparse_topk_lens=(
                sparse_topk_lens if sparse_topk_lens is not None else swa_topk_lens
            ),
            extra_sparse_indices=extra_sparse_indices,
            extra_sparse_topk_lens=extra_sparse_topk_lens,
            out=out,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            sinks=sinks,
            max_q_len=max_q_len,
            cum_seq_lens_q=cum_seq_lens_q,
            seq_lens=seq_lens,
            kv_layout=kv_layout,
            backend="cake",
        )

    if backend != "cake":
        if (
            swa_topk_lens is not None
            or extra_sparse_indices is not None
            or extra_sparse_topk_lens is not None
        ):
            raise ValueError(
                "swa_topk_lens, extra_sparse_indices, and extra_sparse_topk_lens "
                "are only supported on SM120/SM121 and by backend='cake'"
            )
        if sparse_topk_lens_offset != 0:
            raise ValueError(
                "sparse_topk_lens_offset is only supported by backend='cake'"
            )
        if sparse_topk_lens is None or compressed_kv_cache is None or seq_lens is None:
            raise ValueError(
                f"backend={backend!r} requires compressed_kv_cache, "
                "sparse_topk_lens, and seq_lens"
            )
        cake_lens = sparse_topk_lens
    else:
        if swa_topk_lens is not None:
            raise ValueError("backend='cake' does not accept swa_topk_lens")
        if isinstance(sparse_topk_lens_offset, bool) or not isinstance(
            sparse_topk_lens_offset, int
        ):
            raise TypeError("sparse_topk_lens_offset must be an int")
        if extra_sparse_topk_lens is not None and extra_sparse_indices is None:
            raise ValueError("extra_sparse_topk_lens requires extra_sparse_indices")
        if extra_sparse_topk_lens is not None and sparse_topk_lens is not None:
            raise ValueError(
                "backend='cake' takes either sparse_topk_lens (counting the 128 "
                "SWA slots) or extra_sparse_topk_lens (compressed slots only), "
                "not both"
            )
        cake_lens = (
            sparse_topk_lens if sparse_topk_lens is not None else extra_sparse_topk_lens
        )
        if cake_lens is None or compressed_kv_cache is None or seq_lens is None:
            raise ValueError(
                "backend='cake' requires compressed_kv_cache, seq_lens, and "
                "sparse_topk_lens or extra_sparse_topk_lens"
            )
    if sparse_indices is None:
        raise ValueError(f"backend={backend!r} requires sparse_indices")
    if multi_ctas_kv_counter_buffer is not None and backend != "trtllm-gen":
        raise ValueError(
            "multi_ctas_kv_counter_buffer is only used by backend='trtllm-gen'"
        )

    # ``cake_lens`` is whichever length tensor the caller supplied; remember
    # whether it was the combined table so the cake host receives exactly one
    # of sparse_topk_lens / extra_sparse_topk_lens.
    combined_lens_given = sparse_topk_lens is not None
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
        cake_lens,
        None if rope_quant else out,
        sinks,
        kv_layout,
        cum_seq_lens_q,
        max_q_len,
        allow_sm120_packed_kv=False,
        metadata_rows_may_be_fewer=backend in ("cake", "trtllm-gen"),
        check_topk_lens_range=backend != "cake",
    )

    if rope_quant:
        num_tokens, num_heads = query_flat.shape[:2]
        if out is None:
            out = _allocate_dsv4_rope_quant_output(num_tokens, num_heads, query.device)
        if dsv4_output_scale is None:
            dsv4_output_scale = _allocate_dsv4_rope_quant_output_scale(
                num_tokens, num_heads, query.device
            )
    elif out is None and backend != "cake":
        # backend='cake' allocates in run_cake_dsv4 once the route is known
        # (route-dependent output base phase, see cake_dsv4._OUT_PHASE_BY_ROUTE).
        out = torch.empty(expected_out_shape, dtype=torch.bfloat16, device=query.device)

    check_shape_dtype_device(
        seq_lens, (batch_size,), torch.int32, query.device, "seq_lens"
    )
    validate_sync_inputs = _validate_dsv4_sync_checks(query.device)
    if validate_sync_inputs:
        if cum_seq_lens_q is None:
            q_lens = seq_lens.new_full((batch_size,), q_len_per_request)
        else:
            q_lens = cum_seq_lens_q[1:] - cum_seq_lens_q[:-1]
        if torch.any(seq_lens < q_lens).item():
            raise ValueError(
                "seq_lens must be greater than or equal to the per-request query "
                "lengths so TRTLLM-GEN can derive the SWA-128 valid window"
            )
    if (
        validate_sync_inputs
        and dsv4_inv_rope_cos_sin_cache is not None
        and dsv4_inv_rope_cos_sin_cache.ndim == 2
        and torch.any(seq_lens > dsv4_inv_rope_cos_sin_cache.size(0)).item()
    ):
        raise ValueError(
            "dsv4_inv_rope_cos_sin_cache does not cover all derived query positions"
        )

    if backend == "cake":
        from .cake_dsv4 import run_cake_dsv4

        if extra_sparse_indices is not None:
            check_shape_dtype_device(
                extra_sparse_indices,
                (sparse_indices.shape[0], extra_sparse_indices.shape[-1]),
                torch.int32,
                query.device,
                "extra_sparse_indices",
            )
        return run_cake_dsv4(
            query=query_flat,
            swa_kv_cache=swa_kv_cache,
            compressed_kv_cache=compressed_kv_cache,
            workspace_buffer=workspace_buffer,
            sparse_indices=sparse_indices,
            # Combined-convention lengths, or None when the compressed-only
            # extra_sparse_topk_lens carries them (the checked tensor above is
            # the extra table in that case and must not be passed twice).
            sparse_topk_lens=sparse_topk_lens if combined_lens_given else None,
            extra_sparse_indices=extra_sparse_indices,
            extra_sparse_topk_lens=extra_sparse_topk_lens,
            sparse_topk_lens_offset=sparse_topk_lens_offset,
            out=out,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            sinks=sinks,
            max_q_len=q_len_per_request,
            cum_seq_lens_q=cum_seq_lens_q,
            seq_lens=seq_lens,
            backend="cake",
            out_shape=tuple(expected_out_shape),
            enable_pdl=bool(enable_pdl),
        )

    primary_kv_cache = compressed_kv_cache
    # Padded DP/MTP batches: ``query``/``out`` may carry more rows than the
    # metadata tables. Only the metadata rows are attended; the cubin never sees
    # the padded rows and never writes their output (flashinfer#4671, P1).
    metadata_rows = sparse_indices.shape[0]
    if metadata_rows != query_flat.size(0):
        if rope_quant:
            raise ValueError(
                "padded query rows are not supported together with "
                "dsv4_inv_rope_cos_sin_cache"
            )
        if cum_seq_lens_q is None:
            if metadata_rows % q_len_per_request:
                raise ValueError(
                    "dense padded queries must pad whole requests: "
                    f"{metadata_rows} metadata rows are not a multiple of "
                    f"q_len {q_len_per_request}"
                )
            batch_size = metadata_rows // q_len_per_request
        else:
            # Ragged padding drops trailing requests: the kernel takes its
            # token count from cum_seq_lens_q, so the offsets are cut at the
            # request boundary that equals the metadata row count.
            offsets = cum_seq_lens_q.tolist()
            if metadata_rows not in offsets[1:]:
                raise ValueError(
                    "ragged padded queries must pad whole requests: no "
                    f"cum_seq_lens_q boundary equals the {metadata_rows} metadata rows"
                )
            batch_size = offsets.index(metadata_rows)
            cum_seq_lens_q = cum_seq_lens_q[: batch_size + 1]
        if seq_lens.numel() > batch_size:
            seq_lens = seq_lens[:batch_size]
        query_flat = query_flat[:metadata_rows]
        out = out.reshape(-1, *out.shape[-2:])[:metadata_rows]
        check_shape_dtype_device(
            seq_lens, (batch_size,), torch.int32, query.device, "seq_lens"
        )
    sparse_indices = sparse_indices.reshape(metadata_rows, -1).contiguous()
    sparse_topk_lens = sparse_topk_lens.contiguous()
    has_strided_pages = any(
        kv_cache.stride(-2) != kv_cache.size(-1)
        or kv_cache.stride(0) != kv_cache.size(-2) * kv_cache.stride(-2)
        for kv_cache in (primary_kv_cache, swa_kv_cache)
    )
    if has_strided_pages and sparse_indices_are_storage_offsets is None:
        raise ValueError(
            "sparse_indices_are_storage_offsets must be set for strided KV pools"
        )
    needs_index_remap = has_strided_pages and not sparse_indices_are_storage_offsets
    if remapped_sparse_indices_buffer is not None:
        check_shape_dtype_device(
            remapped_sparse_indices_buffer,
            tuple(sparse_indices.shape),
            torch.int32,
            query.device,
            "remapped_sparse_indices_buffer",
        )
        if not remapped_sparse_indices_buffer.is_contiguous():
            raise ValueError("remapped_sparse_indices_buffer must be contiguous")
        if (
            needs_index_remap
            and remapped_sparse_indices_buffer.untyped_storage().data_ptr()
            == sparse_indices.untyped_storage().data_ptr()
        ):
            raise ValueError(
                "remapped_sparse_indices_buffer must not share storage with "
                "sparse_indices"
            )
    elif needs_index_remap:
        remapped_sparse_indices_buffer = torch.empty_like(sparse_indices)
    else:
        remapped_sparse_indices_buffer = sparse_indices

    op = get_trtllm_gen_fmha_module()
    run_func = getattr(op, "trtllm_paged_attention_decode_sparse_mla_dsv4", None)
    if run_func is None:
        raise RuntimeError(
            "trtllm_paged_attention_decode_sparse_mla_dsv4 is not available. "
            "Rebuild FlashInfer with the DeepSeek V4 sparse MLA TRTLLM-GEN launcher."
        )

    sm_count = get_device_sm_count(query.device)
    # Caller-owned buffer when provided (validated against the required size);
    # otherwise a fresh zero-initialized buffer. The kernel self-resets the
    # counters at the end of the launch, so no explicit re-zeroing is required.
    multi_ctas_kv_counter_buffer = _resolve_trtllm_gen_multi_ctas_kv_counter_buffer(
        multi_ctas_kv_counter_buffer,
        batch_size,
        query_flat.size(1),
        sm_count,
        query.device,
    )
    run_func(
        out,
        query_flat,
        primary_kv_cache,
        swa_kv_cache,
        workspace_buffer,
        multi_ctas_kv_counter_buffer,
        sparse_indices,
        remapped_sparse_indices_buffer,
        sparse_indices_are_storage_offsets is True,
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
        dsv4_inv_rope_cos_sin_cache,
        dsv4_output_scale,
    )
    if rope_quant:
        assert dsv4_output_scale is not None
        return out, dsv4_output_scale
    return out


# Keep the backend-neutral spelling as a compatibility alias, while the
# existing TRTLLM-prefixed API remains the canonical public callable.
batch_decode_sparse_mla_dsv4 = trtllm_batch_decode_sparse_mla_dsv4


# Backend dependencies for functional dispatch and public API exports.
from ._batch_mla._backends.trtllm_gen_backend import (
    _BatchMLAPagedAttentionTrtllmGenBackend,
    get_trtllm_gen_fmha_module,
    _prepare_trtllm_gen_mla_output,
    _trtllm_gen_mla_incompatibility_reason,
)
from ._batch_mla._backends._fa_common import (
    get_batch_mla_module as get_batch_mla_module,
)
from ._batch_mla._backends.cute_dsl_modular_backend import (
    _BatchMLAPagedAttentionCuteDslModularBackend,
)
from ._batch_mla._backends.cute_dsl_monolithic_backend import (
    _BatchMLAPagedAttentionCuteDslMonolithicBackend,
)
from ._batch_mla._backends.xqa_backend import _BatchMLAPagedAttentionXqaBackend

_XQA_MLA_TUNING_CONFIG = TuningConfig(use_cuda_graph=True)
from ._batch_mla._backends.cutlass_backend import get_mla_module as get_mla_module
from ._batch_mla._contracts import MLAPlanMetadata as MLAPlanMetadata
from ._batch_mla._wrapper import (
    BatchMLAPagedAttentionWrapper as BatchMLAPagedAttentionWrapper,
)

# ---------------------------------------------------------------------------
# Autotuning support for trtllm_batch_decode_with_kv_cache_mla
# ---------------------------------------------------------------------------

# Keep the trtllm-gen autotune sweep bounded; actual counter storage is sized
# dynamically per profiled batch.
_TRTLLM_GEN_MLA_MAX_BATCH = 8192

# Query rows (batch * q_len) up to which the Cake candidate is profiled and
# runs. It keeps every generic CLC launch within the device work-order window.
# The FP8 page-64 route allocates its split-K partials on every call (about
# rows * num_split * 129 KiB, num_split 2 from 37 rows on), also inside every
# captured graph, so FP8 stops at 128 rows (about 33 MiB). The BF16 split-KV
# route allocates its partials on every call too, at most one wave of clusters
# (rows * num_split <= num_sms // 2) of 128.5 KiB each, about 10 MiB. The other
# routes the candidate takes allocate nothing per call but an 8 KiB work order.
_CAKE_MLA_MAX_ROWS = 512
_CAKE_MLA_FP8_MAX_ROWS = 128


# The units for which ``backend="auto"`` offers the Cake candidate, per device.
#
# Key: (arch, SM count, dtype name, page size, q_len, has_sinks), the exact
# device (e.g. ("sm_100a", 148) for B200, ("sm_103a", 152) for GB300).
# Value: (domain, max_seq_len bucket, first batch, last batch) ranges. A range is
# a union of whole tuning buckets of ``_compute_mla_decode_buckets``, every batch
# in it launches ``domain``, and the max_seq_len bucket is
# ``_round_to_seq_len_bucket(max_seq_len)``.
#
# The table is derived mechanically from explicit measurements on each listed
# device, using that device's measurements only. A unit is listed only when
# every planned cell has at least three rounds and, per cell, the median of
# cake / best_other is at most 0.98 and every round at most 1.00, where
# best_other is the faster of tuned CuTe DSL and TRTLLM-GEN. The planned cells
# of a unit are the first, middle and last batch of its buckets; KV lengths of
# the lower bucket edge plus one page, three quarters of the bucket and the
# upper edge; and the uniform, varied, padded and short length patterns (short:
# every row at most a quarter of the bucket while max_seq_len is its upper
# edge).
#
# Devices, dtypes, page sizes, q_len values, max_seq_len buckets and batches
# that are not listed never offer Cake. Requests with sinks are not offered.
_CAKE_MLA_AUTO_OFFERS: Dict[
    Tuple[str, int, str, int, int, bool], Tuple[Tuple[str, int, int, int], ...]
] = {
    ("sm_100a", 148, "bfloat16", 32, 1, False): (
        ("mla_bf16_clc", 256, 128, 255),
        ("mla_bf16_clc", 512, 128, 255),
        ("mla_bf16_clc", 1024, 128, 255),
        ("mla_bf16_clc", 2048, 128, 255),
        ("mla_bf16_vquarter", 128, 1, 15),
        ("mla_bf16_vquarter", 256, 1, 15),
        ("mla_bf16_vquarter", 512, 1, 15),
    ),
    ("sm_100a", 148, "bfloat16", 32, 2, False): (
        ("mla_bf16_clc", 512, 64, 127),
        ("mla_bf16_clc", 1024, 64, 127),
        ("mla_bf16_clc", 2048, 64, 255),
        ("mla_bf16_vquarter", 128, 1, 7),
        ("mla_bf16_vquarter", 256, 1, 7),
        ("mla_bf16_vquarter", 512, 1, 7),
    ),
    ("sm_100a", 148, "bfloat16", 64, 1, False): (
        ("mla_bf16_vquarter", 128, 1, 15),
        ("mla_bf16_vquarter", 256, 1, 15),
        ("mla_bf16_vquarter", 512, 1, 15),
    ),
    ("sm_100a", 148, "bfloat16", 64, 2, False): (
        ("mla_bf16_vquarter", 128, 1, 7),
        ("mla_bf16_vquarter", 256, 1, 7),
        ("mla_bf16_vquarter", 512, 1, 7),
    ),
    ("sm_100a", 148, "float8_e4m3fn", 64, 1, False): (
        ("mla_fp8_page64_pdl", 2048, 16, 31),
        ("mla_fp8_page64_pdl", 4096, 16, 31),
        ("mla_fp8_page64_pdl", 8192, 16, 31),
    ),
    ("sm_100a", 148, "float8_e4m3fn", 64, 2, False): (
        ("mla_fp8_page64_pdl", 2048, 8, 15),
        ("mla_fp8_page64_pdl", 4096, 8, 15),
        ("mla_fp8_page64_pdl", 8192, 8, 15),
        ("mla_fp8_page64_pdl", 16384, 8, 15),
    ),
    ("sm_103a", 152, "bfloat16", 32, 1, False): (
        ("mla_bf16_clc", 1024, 128, 255),
        ("mla_bf16_clc", 2048, 128, 511),
        ("mla_bf16_clc", 4096, 128, 511),
        ("mla_bf16_vquarter", 128, 1, 15),
        ("mla_bf16_vquarter", 256, 1, 15),
        ("mla_bf16_vquarter", 512, 1, 15),
        ("mla_bf16_vquarter", 1024, 8, 15),
    ),
    ("sm_103a", 152, "bfloat16", 32, 2, False): (
        ("mla_bf16_clc", 1024, 64, 127),
        ("mla_bf16_clc", 2048, 64, 255),
        ("mla_bf16_clc", 4096, 128, 255),
        ("mla_bf16_vquarter", 128, 1, 7),
        ("mla_bf16_vquarter", 256, 1, 7),
        ("mla_bf16_vquarter", 512, 1, 7),
    ),
    ("sm_103a", 152, "bfloat16", 64, 1, False): (
        ("mla_bf16_vquarter", 128, 1, 15),
        ("mla_bf16_vquarter", 256, 1, 15),
        ("mla_bf16_vquarter", 512, 1, 15),
        ("mla_bf16_vquarter", 1024, 8, 15),
    ),
    ("sm_103a", 152, "bfloat16", 64, 2, False): (
        ("mla_bf16_vquarter", 128, 1, 7),
        ("mla_bf16_vquarter", 256, 1, 7),
        ("mla_bf16_vquarter", 512, 1, 7),
        ("mla_bf16_vquarter", 1024, 4, 7),
    ),
    ("sm_103a", 152, "float8_e4m3fn", 64, 1, False): (
        ("mla_fp8_page64_pdl", 2048, 16, 31),
    ),
    ("sm_103a", 152, "float8_e4m3fn", 64, 2, False): (
        ("mla_fp8_page64_pdl", 1024, 2, 3),
        ("mla_fp8_page64_pdl", 2048, 8, 15),
        ("mla_fp8_page64_pdl", 4096, 8, 15),
    ),
}


def _cake_mla_dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).rsplit(".", 1)[-1]


def _cake_mla_auto_offer_ranges(
    arch: str,
    num_sms: int,
    dtype: torch.dtype,
    page_size: int,
    q_len: int,
    has_sinks: bool,
    max_seq_len: int,
) -> Tuple[Tuple[str, int, int], ...]:
    """The ``(domain, first batch, last batch)`` ranges ``_CAKE_MLA_AUTO_OFFERS``
    lists for this device, request configuration and max_seq_len bucket."""
    if has_sinks:
        return ()
    entries = _CAKE_MLA_AUTO_OFFERS.get(
        (arch, num_sms, _cake_mla_dtype_name(dtype), page_size, q_len, has_sinks),
        (),
    )
    seq_bucket = _round_to_seq_len_bucket(max_seq_len)
    return tuple(
        (domain, first, last)
        for domain, bucket, first, last in entries
        if bucket == seq_bucket
    )


def _cake_mla_auto_offered(
    arch: str,
    num_sms: int,
    dtype: torch.dtype,
    page_size: int,
    domain: str,
    q_len: int,
    has_sinks: bool,
    max_seq_len: int,
    batch_size: int,
) -> bool:
    """Whether ``backend="auto"`` offers the Cake candidate for a batch of
    ``batch_size`` requests launching ``domain`` on this device: only for the
    units ``_CAKE_MLA_AUTO_OFFERS`` lists, where Cake measured faster than the
    other candidates. Explicit ``backend="cake"`` is unaffected."""
    return any(
        listed == domain and first <= batch_size <= last
        for listed, first, last in _cake_mla_auto_offer_ranges(
            arch, num_sms, dtype, page_size, q_len, has_sinks, max_seq_len
        )
    )


# Cake MLA families the candidate loaded (it never builds one during stream
# capture), and those whose JIT build or load failed in this process: the
# candidate does not retry them.
_CAKE_MLA_LOADED_FAMILIES: set = set()
_CAKE_MLA_UNAVAILABLE_FAMILIES: set = set()


@functools.cache
def _cake_mla_module():
    """``flashinfer.mla.cake_trtllm_mla_blackwell``, imported on first use."""
    from . import cake_trtllm_mla_blackwell

    return cake_trtllm_mla_blackwell


# Host facts of the Cake candidate that depend only on scalars, on the offer
# table and on the split-KV device ranges: per (arch, SM count, dtype, page
# size, q_len, max_seq_len bucket) the offered batch ranges, and per request
# sizes and batch the launched families (keyed only by the parts of
# max_seq_len they read, and by the device's split-KV range, which decides the
# domain). They are derived from the objects ``_CAKE_MLA_TABLE_MEMO_OWNER``
# holds (the offer table and the Cake module's ``_SPLITKV_RANGE``) and are
# dropped when either name another object. Bounded: cleared when it grows past
# ``_CAKE_MLA_MEMO_LIMIT`` entries.
_CAKE_MLA_TABLE_MEMO: Dict[Any, Any] = {}
_CAKE_MLA_TABLE_MEMO_OWNER: List[Any] = [None]
_CAKE_MLA_MEMO_LIMIT = 4096
_MISSING = object()
# ``(device facts function, device index)`` of the devices without Cake
# programs (not SM100a/SM103a): ``_device_facts`` caches only its result, so
# without this an auto call there would pay a capability query and a raise.
_CAKE_MLA_NO_PROGRAM_DEVICES: Set[Tuple[Any, int]] = set()


def _cake_mla_table_memo() -> Dict[Any, Any]:
    table = _CAKE_MLA_AUTO_OFFERS
    split_ranges = _cake_mla_module()._SPLITKV_RANGE
    owner = _CAKE_MLA_TABLE_MEMO_OWNER[0]
    if owner is None or owner[0] is not table or owner[1] is not split_ranges:
        _CAKE_MLA_TABLE_MEMO.clear()
        _CAKE_MLA_TABLE_MEMO_OWNER[0] = (table, split_ranges)
    elif len(_CAKE_MLA_TABLE_MEMO) > _CAKE_MLA_MEMO_LIMIT:
        _CAKE_MLA_TABLE_MEMO.clear()
    return _CAKE_MLA_TABLE_MEMO


# The results of ``_cake_mla_auto_batch_ranges`` per request configuration
# ``(device index, dtype, page size, q_len, max_seq_len bucket)``: ``(offer
# table, device facts function, {configuration: ranges})``, replaced as one tuple
# when the table or the Cake module's ``_device_facts`` is another object
# (tests patch both). A configuration seen before then pays one dict lookup
# and two identity checks instead of the device facts and the table memo, also
# when configurations alternate. Bounded like ``_CAKE_MLA_TABLE_MEMO``.
_CAKE_MLA_RANGES_MEMO: List[Any] = [None]


def _cake_mla_auto_batch_ranges(
    device_index: int,
    dtype: torch.dtype,
    page_size: int,
    q_len: int,
    has_sinks: bool,
    max_seq_len: int,
) -> Tuple[Tuple[int, int], ...]:
    """The ``(first batch, last batch)`` ranges ``_CAKE_MLA_AUTO_OFFERS`` lists
    for this request configuration on the device of ``device_index`` (any
    domain), memoized: ``()`` for an unlisted unit or a device that is not
    SM100a/SM103a. A configuration's result is reused while the offer table
    and the Cake module's ``_device_facts`` are the same objects (the ranges
    read nothing else, and only the bucket of ``max_seq_len``, so every
    ``max_seq_len`` of a bucket shares one entry)."""
    if has_sinks:
        return ()
    seq_bucket = _round_to_seq_len_bucket(max_seq_len)
    configuration = (device_index, dtype, page_size, q_len, seq_bucket)
    device_facts = _cake_mla_module()._device_facts
    table = _CAKE_MLA_AUTO_OFFERS
    memo = _CAKE_MLA_RANGES_MEMO[0]
    if memo is not None and memo[0] is table and memo[1] is device_facts:
        ranges = memo[2].get(configuration)
        if ranges is not None:
            return ranges
    else:
        memo = None
    ranges = _cake_mla_lookup_batch_ranges(
        device_facts, device_index, dtype, page_size, q_len, seq_bucket
    )
    if _CAKE_MLA_AUTO_OFFERS is table:  # else replaced during the lookup
        if memo is None:
            memo = (table, device_facts, {})
            _CAKE_MLA_RANGES_MEMO[0] = memo
        elif len(memo[2]) > _CAKE_MLA_MEMO_LIMIT:
            memo[2].clear()
        memo[2][configuration] = ranges
    return ranges


def _cake_mla_lookup_batch_ranges(
    device_facts: Callable[[int], Tuple[str, int]],
    device_index: int,
    dtype: torch.dtype,
    page_size: int,
    q_len: int,
    seq_bucket: int,
) -> Tuple[Tuple[int, int], ...]:
    """``_cake_mla_auto_batch_ranges`` without sinks for the ``max_seq_len``
    bucket ``seq_bucket``, from the table memo."""
    if (device_facts, device_index) in _CAKE_MLA_NO_PROGRAM_DEVICES:
        return ()
    try:
        arch, num_sms = device_facts(device_index)
    except RuntimeError:  # not SM100a/SM103a: no Cake programs
        _CAKE_MLA_NO_PROGRAM_DEVICES.add((device_facts, device_index))
        return ()
    memo = _cake_mla_table_memo()
    key = ("ranges", arch, num_sms, dtype, page_size, q_len, seq_bucket)
    ranges = memo.get(key)
    if ranges is None:
        ranges = tuple(
            (first, last)
            for _, first, last in _cake_mla_auto_offer_ranges(
                arch, num_sms, dtype, page_size, q_len, False, seq_bucket
            )
        )
        memo[key] = ranges
    return ranges


def _in_batch_ranges(batch_size: int, ranges: Tuple[Tuple[int, int], ...]) -> bool:
    for first, last in ranges:  # noqa: SIM110 - no generator on the per-call path
        if first <= batch_size <= last:
            return True
    return False


def _resolve_cute_dsl_workspace_sizer(
    cute_dsl_impl: str,
    sinks: Optional[Union[List[torch.Tensor], Tuple[torch.Tensor, ...], torch.Tensor]],
    enable_dcp: bool = False,
):
    """Resolve the selected CuTeDSL implementation and its workspace policy."""
    from ..cute_dsl.attention.mla_dispatch import _resolve_impl

    resolved_impl = _resolve_impl(
        requested=cute_dsl_impl,
        kwargs={"sinks": sinks, "enable_dcp": enable_dcp},
    )
    if resolved_impl == "monolithic":
        from ..cute_dsl.attention.monolithic.mla_decode import (
            _get_split_kv_and_workspace_size as monolithic_workspace_sizer,
        )

        return monolithic_workspace_sizer, resolved_impl

    from ..cute_dsl.attention.wrappers.batch_mla import (
        _get_split_kv_and_workspace_size as modular_workspace_sizer,
    )

    return modular_workspace_sizer, resolved_impl


def _get_cute_dsl_workspace_sizer(
    cute_dsl_impl: str,
    sinks: Optional[Union[List[torch.Tensor], Tuple[torch.Tensor, ...], torch.Tensor]],
    enable_dcp: bool = False,
):
    """Return the workspace policy owned by the selected CuTeDSL implementation."""
    return _resolve_cute_dsl_workspace_sizer(cute_dsl_impl, sinks, enable_dcp)[0]


def _call_cute_dsl_workspace_sizer(
    workspace_sizer,
    resolved_impl: str,
    batch_size: int,
    q_len: int,
    num_heads: int,
    kv_lora_rank: int,
    max_active_blocks: int,
    max_seq_len: int,
):
    """Call an implementation's workspace policy with its supported arguments."""
    args = (batch_size, q_len, num_heads, kv_lora_rank, max_active_blocks)
    if resolved_impl == "monolithic":
        return workspace_sizer(*args, max_seq_len)
    return workspace_sizer(*args)


def _cute_dsl_max_supported_batch(
    workspace_bytes: int,
    q_len: int,
    num_heads: int,
    kv_lora_rank: int,
    max_active_blocks: int,
    max_seq_len: int,
    candidate_max: int,
    cute_dsl_impl: str,
    sinks: Optional[Union[List[torch.Tensor], Tuple[torch.Tensor, ...], torch.Tensor]],
    enable_dcp: bool = False,
) -> int:
    """Largest batch the caller's workspace can support for cute-dsl MLA decode.

    Cute-dsl's workspace requirement grows with B via the kernel's per-CTA
    split-K state. Binary-search for the largest ``B <= candidate_max`` whose
    ``get_workspace_size(...)`` fits in ``workspace_bytes``. The search is
    memoized per workspace policy and sizes (``_cute_dsl_max_batch_search``);
    only the monolithic policy reads ``max_seq_len``, so the others' entries
    are shared by every ``max_seq_len``.
    """
    workspace_sizer, resolved_impl = _resolve_cute_dsl_workspace_sizer(
        cute_dsl_impl, sinks, enable_dcp
    )
    return _cute_dsl_max_batch_search(
        workspace_sizer,
        resolved_impl,
        workspace_bytes,
        q_len,
        num_heads,
        kv_lora_rank,
        max_active_blocks,
        max_seq_len if resolved_impl == "monolithic" else 0,
        candidate_max,
    )


@functools.lru_cache(maxsize=256)
def _cute_dsl_max_batch_search(
    workspace_sizer,
    resolved_impl: str,
    workspace_bytes: int,
    q_len: int,
    num_heads: int,
    kv_lora_rank: int,
    max_active_blocks: int,
    max_seq_len: int,
    candidate_max: int,
) -> int:
    """The binary search of ``_cute_dsl_max_supported_batch``. The workspace
    policies are pure functions of these arguments."""
    lo, hi = 1, max(1, candidate_max)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        _, ws = _call_cute_dsl_workspace_sizer(
            workspace_sizer,
            resolved_impl,
            mid,
            q_len,
            num_heads,
            kv_lora_rank,
            max_active_blocks,
            max_seq_len,
        )
        if ws <= workspace_bytes:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _compute_mla_decode_buckets(
    workspace_buffer: torch.Tensor,
    runner_names: Sequence[str],
    q_len: int,
    num_heads: int,
    kv_lora_rank: int,
    max_seq_len: int,
    device: torch.device,
    cute_dsl_impl: str,
    sinks: Optional[Union[List[torch.Tensor], Tuple[torch.Tensor, ...], torch.Tensor]],
    enable_dcp: bool = False,
) -> Tuple[int, ...]:
    """Compute the autotune bucket list from kernel/workspace limits only.

    The cap is intentionally independent of the caller's ``query.shape[0]``
    and ``kv_cache.shape[0]`` so that autotuning at any single batch size
    still tunes the full range a runner can actually serve. Buckets above a
    runner's individual cap are handled by that runner's ``get_valid_tactics``
    returning ``[]`` for the affected profiles.

    Caps:
      - trtllm-gen kernel-hardcoded ceiling: 8192
      - cute-dsl dynamic workspace cap (via binary search, at most 8192)

    With trtllm-gen listed the cap is 8192 whatever the cute-dsl cap is, so
    the cute-dsl workspace search is skipped.
    """
    if "trtllm-gen" in runner_names:
        return _mla_decode_hybrid_buckets(_TRTLLM_GEN_MLA_MAX_BATCH)
    cap = 0
    if "cute-dsl" in runner_names:
        from ..cute_dsl.utils import get_num_sm

        cute_dsl_cap = _cute_dsl_max_supported_batch(
            workspace_bytes=workspace_buffer.numel() * workspace_buffer.element_size(),
            q_len=q_len,
            num_heads=num_heads,
            kv_lora_rank=kv_lora_rank,
            max_active_blocks=get_num_sm(device),
            max_seq_len=max_seq_len,
            candidate_max=_TRTLLM_GEN_MLA_MAX_BATCH,
            cute_dsl_impl=cute_dsl_impl,
            sinks=sinks,
            enable_dcp=enable_dcp,
        )
        cap = max(cap, cute_dsl_cap)

    return _mla_decode_hybrid_buckets(max(1, cap))


@functools.lru_cache(maxsize=64)
def _mla_decode_hybrid_buckets(cap: int) -> Tuple[int, ...]:
    """``get_hybrid_num_tokens_buckets(cap)``, memoized (a tuple)."""
    from ..fused_moe.utils import get_hybrid_num_tokens_buckets

    return get_hybrid_num_tokens_buckets(cap)


def _cake_trtllm_mla_blackwell_supports(
    query: torch.Tensor,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    sparse_mla_top_k: int,
) -> bool:
    """Whether the generated TRT-LLM-style Blackwell MLA programs cover this dimension tuple."""
    from .cake_trtllm_mla_blackwell import supports_dimension_tuple

    return supports_dimension_tuple(
        qk_nope_head_dim,
        kv_lora_rank,
        qk_rope_head_dim,
        int(query.shape[-2]),
        int(sparse_mla_top_k),
    )


def _cake_mla_auto_candidate_reason(
    *,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    out: torch.Tensor,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    sparse_mla_top_k: int,
    bmm1_scale: Union[float, torch.Tensor],
    bmm2_scale: Union[float, torch.Tensor],
    sinks: Optional[List[torch.Tensor]],
    skip_softmax_threshold_scale_factor: Optional[float],
    uses_shared_paged_kv_idx: bool,
    lse_requested: bool,
    use_fp16_softmax: Optional[bool],
    multi_ctas_kv_counter_buffer: Optional[torch.Tensor],
    sparse_mla_top_k_lens: Optional[torch.Tensor],
    enable_dcp: bool,
) -> Optional[str]:
    """Why the Cake Blackwell MLA route cannot join uniform ``backend="auto"`` tuning.

    Returns ``None`` when the route serves this request with the semantics of the
    TRTLLM-GEN / CuTe DSL candidates. The switch, process-group and tuner gate is
    :func:`~flashinfer.autotuner.cake_auto_candidate_open`; batch-dependent limits
    are checked per profile by :meth:`_CakeTrtllmMlaBlackwellRunner.get_valid_tactics`
    and per call by :meth:`_CakeTrtllmMlaBlackwellRunner.validate_tactic`. The
    checks on scalars, dtypes and sizes are memoized
    (``_cake_mla_auto_static_reason``); the sink tensor and the contiguity of
    every tensor are checked on every call.
    """
    reason = _cake_mla_auto_scalar_reason(
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        sparse_mla_top_k=sparse_mla_top_k,
        sparse_mla_top_k_lens=sparse_mla_top_k_lens,
        skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
        lse_requested=lse_requested,
        use_fp16_softmax=use_fp16_softmax,
        enable_dcp=enable_dcp,
        multi_ctas_kv_counter_buffer=multi_ctas_kv_counter_buffer,
        uses_shared_paged_kv_idx=uses_shared_paged_kv_idx,
    )
    if reason is not None:
        return reason
    reason = _cake_mla_auto_static_reason(
        get_compute_capability(query.device),
        query.dtype,
        kv_cache.dtype,
        kv_cache.shape[-2],
        bmm1_scale,
        bmm2_scale,
        seq_lens.dtype,
        block_tables.dtype,
        qk_nope_head_dim,
        kv_lora_rank,
        qk_rope_head_dim,
        query.size(-2),
        sinks is not None,
    )
    if reason is not None:
        return reason
    if sinks is not None:
        sink = (
            sinks[0] if isinstance(sinks, (list, tuple)) and len(sinks) == 1 else sinks
        )
        if not (
            isinstance(sink, torch.Tensor)
            and sink.device == query.device
            and sink.dtype == torch.float32
            and sink.is_contiguous()
            and tuple(sink.shape) == (query.size(-2),)
        ):
            return _CAKE_MLA_SINK_REASON
    if not (
        query.is_contiguous()
        and kv_cache.is_contiguous()
        and block_tables.is_contiguous()
        and seq_lens.is_contiguous()
        and out.is_contiguous()
    ):
        return _CAKE_MLA_CONTIGUITY_REASON
    return None


# The reasons of ``_cake_mla_auto_candidate_reason`` that depend on more than
# scalars, dtypes and sizes (the sink tensor, strides). Every other reason is
# a function of the dispatcher's memo key (``_CAKE_MLA_AUTO_MEMO``).
_CAKE_MLA_SINK_REASON = "requires one contiguous FP32 sink value per head"
_CAKE_MLA_CONTIGUITY_REASON = (
    "requires contiguous query, kv_cache, block_tables, seq_lens and out"
)


def _cake_mla_auto_scalar_reason(
    *,
    bmm1_scale: Union[float, torch.Tensor],
    bmm2_scale: Union[float, torch.Tensor],
    sparse_mla_top_k: int,
    sparse_mla_top_k_lens: Optional[torch.Tensor],
    skip_softmax_threshold_scale_factor: Optional[float],
    lse_requested: bool,
    use_fp16_softmax: Optional[bool],
    enable_dcp: bool,
    multi_ctas_kv_counter_buffer: Optional[torch.Tensor],
    uses_shared_paged_kv_idx: bool,
) -> Optional[str]:
    """The checks of ``_cake_mla_auto_candidate_reason`` on arguments other
    than the tensors' dtypes, sizes and strides; ``None`` when they pass. The
    dispatcher runs them before the Cake candidate gate and the memo key, so
    such calls pay for neither."""
    if isinstance(bmm1_scale, torch.Tensor) or isinstance(bmm2_scale, torch.Tensor):
        return "requires host float bmm1_scale / bmm2_scale"
    # The sparse route reads every index (no -1 padding), the route has no
    # skip-softmax and its LSE units do not follow return_lse_base.
    if sparse_mla_top_k > 0 or sparse_mla_top_k_lens is not None:
        return "does not join sparse MLA selection"
    if skip_softmax_threshold_scale_factor is not None:
        return "does not implement skip-softmax"
    if lse_requested:
        return "does not join selection when LSE is requested"
    if use_fp16_softmax:
        return "does not implement use_fp16_softmax"
    if enable_dcp or multi_ctas_kv_counter_buffer is not None:
        return "does not support DCP or multi_ctas_kv_counter_buffer"
    if not uses_shared_paged_kv_idx:
        return "requires uses_shared_paged_kv_idx=True"
    return None


@functools.lru_cache(maxsize=1024)
def _cake_mla_auto_static_reason(
    capability: Tuple[int, int],
    dtype: torch.dtype,
    kv_dtype: torch.dtype,
    page_size: int,
    bmm1_scale: float,
    bmm2_scale: float,
    seq_lens_dtype: torch.dtype,
    block_tables_dtype: torch.dtype,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    num_heads: int,
    has_sinks: bool,
) -> Optional[str]:
    """The checks of ``_cake_mla_auto_candidate_reason`` on the device's
    compute capability, the dtypes, sizes and host scales; ``None`` when they
    pass."""
    if capability not in ((10, 0), (10, 3)):
        return "requires SM100a or SM103a"
    if dtype not in (torch.bfloat16, torch.float8_e4m3fn) or kv_dtype != dtype:
        return "requires BF16 or FP8 E4M3 query and kv_cache of the same dtype"
    if page_size not in (32, 64):
        return "requires page size 32 or 64"
    if not (math.isfinite(bmm1_scale) and math.isfinite(bmm2_scale)):
        return "requires finite bmm1_scale / bmm2_scale"
    if seq_lens_dtype != torch.int32 or block_tables_dtype != torch.int32:
        return "requires int32 seq_lens and block_tables"
    if not _cake_mla_module().supports_dimension_tuple(
        qk_nope_head_dim, kv_lora_rank, qk_rope_head_dim, num_heads, 0
    ):
        return "has no generated program for this dimension tuple"
    # Other head counts, FP8 page 32 and FP8 with sinks take the generic tail
    # program. It rebuilds its row plan with torch ops on every stateless call,
    # serves max_seq_len <= 4096 only, and on FP8 page 32 runs for milliseconds,
    # so it does not join.
    if num_heads != 128 or (
        dtype == torch.float8_e4m3fn and (page_size != 64 or has_sinks)
    ):
        return "takes the generic tail program, which does not join selection"
    return None


# Outcomes of ``_CakeTrtllmMlaBlackwellRunner.validate_tactic``.
_CAKE_VALID_OK = 0
_CAKE_VALID_STABLE_MISS = 1
_CAKE_VALID_UNREADY = 2


class _CakeTrtllmMlaBlackwellRunner(TunableRunner):
    """Cake Blackwell MLA decode as the challenger of uniform ``backend="auto"``.

    Wraps ``trtllm_mla_blackwell_decode``, the route ``backend="cake"`` takes for
    its generated dimension tuples, over the functional tuner inputs
    ``[query, block_tables, seq_lens, out]``. The route picks its program on the
    host. It is a candidate of the second selection stage
    (``_MLA_CAKE_MARGIN_OP``), enumerated before ``_MlaStageOneWinnerRunner``
    (see ``_mla_cake_stage_two_runners``), which replays the runner and tactic
    the first stage (the TRTLLM-GEN / CuTe DSL selection, unchanged) chose. Its
    single tactic per profile names that first-stage winner, ``(runner class
    name, tactic)``: the profile offers it only when the first stage holds a
    winner there, and a cached Cake choice is valid only while the first stage
    still selects the winner it beat or tied. An untuned call or a
    cache miss never selects it.

    The route runs with ``stateless=True``: like the other candidates it reads
    ``seq_lens`` and ``block_tables`` on the device on every call, so in-place or
    non-torch input updates and CUDA-graph replay with updated inputs stay
    correct, and it keeps no per-identity plan, per-shape scratch or per-stream
    work order that a captured graph could outlive or share. It never reads
    ``seq_lens`` on the host, so only its value-split, generic CLC, FP8 page-64
    and split-KV programs serve it, never its exact-length or generic tail
    programs.
    A padded row (no causal KV token) outputs exact zeros. Per call it launches
    the kernels of explicit ``backend="cake"`` for the same program: the
    attention program, preceded by the longest-first order program when a
    generic CLC launch has more rows than one wave of clusters.

    It serves at most ``_CAKE_MLA_MAX_ROWS`` query rows (FP8:
    ``_CAKE_MLA_FP8_MAX_ROWS``), which bounds the scratch it allocates per call.
    It is offered only for the batches ``_CAKE_MLA_AUTO_OFFERS`` lists for the
    device, where Cake measured faster than the other candidates. A tuning
    bucket offers it only when every batch the bucket serves is listed and
    launches the same programs as the profiled batch, so a measured win holds
    for the whole bucket (a design choice: a bucket whose batches cross a
    program boundary is not offered); never the largest bucket, which also
    serves every larger batch.
    ``validate_tactic`` turns a cached Cake choice into a cache miss when the
    actual batch leaves the listed units, its domains or row cap, when the
    first-stage winner changed, or when a stream capture would have to build
    its program or allocate its FP8 decode table. Every miss, like
    ``FLASHINFER_DISABLE_CAKE_AUTO_CANDIDATES=1``, runs the first stage's
    selection.
    """

    def __init__(
        self,
        *,
        kv_cache: torch.Tensor,
        qk_nope_head_dim: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        max_seq_len: int,
        bmm1_scale: float,
        bmm2_scale: float,
        sinks: Optional[List[torch.Tensor]],
        enable_pdl: bool,
    ) -> None:
        self.kv_cache = kv_cache
        self._qk_nope_head_dim = qk_nope_head_dim
        self._kv_lora_rank = kv_lora_rank
        self._qk_rope_head_dim = qk_rope_head_dim
        self._page_size = int(kv_cache.shape[-2])
        self._max_seq_len = int(max_seq_len)
        self._bmm1_scale = float(bmm1_scale)
        self._bmm2_scale = float(bmm2_scale)
        self.sinks = sinks
        self._enable_pdl = enable_pdl
        # The first-stage winner of the same call (set by the dispatcher).
        self._incumbent: Optional["_MlaStageOneWinnerRunner"] = None
        # The outcome of the last ``validate_tactic``: ``None`` before one,
        # else one of the ``_CAKE_VALID_*`` values, and the families it
        # validated (``_mla_cake_stage_two`` memoizes a decision only when no
        # readiness check decided it).
        self._validation: Optional[int] = None
        self._validated_families: Optional[frozenset] = None
        # validate_tactic calls whose outcome a readiness check decided or that
        # raised.
        self._unready = 0
        # The launch the second stage resolved for this call's inputs
        # (``_launch_plan``); ``None``: ``forward`` takes the
        # dispatcher.
        self._plan: Optional[tuple] = None

    def _domain(
        self,
        inputs,
        num_sms: int,
        batch_size: Optional[int] = None,
        *,
        arch: Optional[str] = None,
    ) -> Optional[str]:
        """The domain the stateless route launches for these inputs (for
        ``batch_size`` requests when given) on a device of ``arch`` with
        ``num_sms`` SMs, or ``None`` when it does not launch within the row cap
        or takes the generic tail program. Without ``arch`` the device-specific
        split-KV domain is never selected (``_families`` and ``_launch_plan``
        pass the device's)."""
        query, block_tables, seq_lens, out = inputs[:4]
        if not (
            query.is_contiguous()
            and block_tables.is_contiguous()
            and seq_lens.is_contiguous()
            and out.is_contiguous()
        ):
            return None
        _, q_len, num_heads, qk_dim = query.shape
        return self._domain_for(
            query.dtype,
            q_len,
            num_heads,
            qk_dim,
            int(block_tables.shape[-1]),
            query.size(0) if batch_size is None else batch_size,
            num_sms,
            arch,
        )

    def _domain_for(
        self,
        dtype: torch.dtype,
        q_len: int,
        num_heads: int,
        qk_dim: int,
        table_width: int,
        batch_size: int,
        num_sms: int,
        arch: Optional[str],
    ) -> Optional[str]:
        """``_domain`` from the request sizes (a pure function of them, of the
        device facts and of the runner's scalars)."""
        cake = _cake_mla_module()
        max_rows = (
            _CAKE_MLA_FP8_MAX_ROWS
            if dtype == torch.float8_e4m3fn
            else _CAKE_MLA_MAX_ROWS
        )
        if batch_size * q_len > max_rows:
            return None
        try:
            domain = cake.select_domain(
                dtype=dtype,
                num_heads=num_heads,
                qk_dim=qk_dim,
                value_dim=self._kv_lora_rank,
                qk_nope_head_dim=self._qk_nope_head_dim,
                page_size=self._page_size,
                topk=0,
                uses_shared_paged_kv_idx=True,
                ragged_query=False,
                enable_sink=self.sinks is not None,
                skip_softmax=False,
                wants_lse=False,
                bmm2_scale=self._bmm2_scale,
                total_q=batch_size * q_len,
                max_seq_len=self._max_seq_len,
                batch_size=batch_size,
                q_len=q_len,
                table_width=table_width,
                kv_lens=None,
                num_sms=num_sms,
                arch=arch,
            )
        except ValueError:
            return None
        return (
            None if domain in (cake.DOMAIN_BF16_TAIL, cake.DOMAIN_FP8_TAIL) else domain
        )

    def _families(self, inputs, batch_sizes) -> Optional[frozenset]:
        """The families the route launches for each of these batch sizes, or
        ``None`` when one of them does not launch, is not a unit
        ``_CAKE_MLA_AUTO_OFFERS`` lists for the device, or launches other
        families than the first.

        Per batch size the result is a pure function of the device facts, the
        request sizes and the runner's scalars, memoized with the offer table
        (``_cake_mla_table_memo``); the contiguity of the inputs is checked on
        every call."""
        cake = _cake_mla_module()
        query, block_tables, seq_lens, out = inputs[:4]
        if not (
            query.is_contiguous()
            and block_tables.is_contiguous()
            and seq_lens.is_contiguous()
            and out.is_contiguous()
        ):
            return None
        try:
            arch, num_sms = cake._device_facts(query.device.index)
        except RuntimeError:
            return None
        _, q_len, num_heads, qk_dim = query.shape
        table_width = int(block_tables.shape[-1])
        splitkv_range = cake._SPLITKV_RANGE.get((arch, num_sms))
        sizes = (
            "families",
            arch,
            num_sms,
            query.dtype,
            q_len,
            num_heads,
            qk_dim,
            table_width,
            self._page_size,
            self._kv_lora_rank,
            self._qk_nope_head_dim,
            self.sinks is not None,
            self._bmm2_scale,
            # The families read max_seq_len only through its bucket (the offer
            # table's units) and whether it reaches the device's split-KV range,
            # which with the range decides whether the split-KV domain serves
            # these sizes; the domain reads it otherwise only to reject the
            # generic tail, which launches nothing here either way.
            _round_to_seq_len_bucket(self._max_seq_len),
            splitkv_range,
            None if splitkv_range is None else self._max_seq_len >= splitkv_range[0],
        )
        memo = _cake_mla_table_memo()
        families = None
        for batch_size in batch_sizes:
            key = (sizes, batch_size)
            launched = memo.get(key, _MISSING)
            if launched is _MISSING:
                launched = self._launched_families(
                    query.dtype,
                    q_len,
                    num_heads,
                    qk_dim,
                    table_width,
                    batch_size,
                    arch,
                    num_sms,
                )
                memo[key] = launched
            if launched is None:
                return None
            if families is None:
                families = launched
            elif launched != families:
                return None
        return families

    def _launched_families(
        self,
        dtype: torch.dtype,
        q_len: int,
        num_heads: int,
        qk_dim: int,
        table_width: int,
        batch_size: int,
        arch: str,
        num_sms: int,
    ) -> Optional[frozenset]:
        """The families a listed batch of ``batch_size`` requests launches, or
        ``None`` (not launched, not listed, or no route)."""
        try:
            domain = self._domain_for(
                dtype, q_len, num_heads, qk_dim, table_width, batch_size, num_sms, arch
            )
            if domain is None or not _cake_mla_auto_offered(
                arch,
                num_sms,
                dtype,
                self._page_size,
                domain,
                q_len,
                self.sinks is not None,
                self._max_seq_len,
                batch_size,
            ):
                return None
            return frozenset(
                _cake_mla_module().launch_families(
                    domain, arch, num_sms, batch_size * q_len
                )
            )
        except (RuntimeError, ValueError):
            return None

    def _launch_plan(self, inputs) -> Optional[tuple]:
        """``(launch, FP8 decode table, sink placeholder, device)`` for these
        inputs: the launch ``trtllm_mla_blackwell_decode(..., stateless=True)``
        makes, resolved once (``DeviceRowLaunch``), or ``None`` when the route
        takes a program whose rows the host plans, or the device has no FP8
        decode table yet. Only for a request without sinks."""
        cake = _cake_mla_module()
        query, block_tables = inputs[0], inputs[1]
        if self.sinks is not None:
            return None
        device = query.device
        lut = cake._FP8_LUTS.get(device)
        if lut is None:
            return None
        arch, num_sms = cake._device_facts(device.index)
        # The domain _families validated: the device's (split-KV included).
        domain = self._domain(inputs, num_sms, arch=arch)
        if domain is None:
            return None
        batch_size, q_len, num_heads, _ = query.shape
        total_q = batch_size * q_len
        launch = cake.device_row_launch(
            domain,
            cake.launch_family(domain, arch, num_sms, total_q),
            arch=arch,
            num_sms=num_sms,
            num_heads=num_heads,
            has_sink=False,
            softmax_scale_log2=self._bmm1_scale * log2e,
            bmm2=self._bmm2_scale,
            total_q=total_q,
            batch_size=batch_size,
            q_len=q_len,
            table_width=int(block_tables.shape[-1]),
            page_size=self._page_size,
            max_seq_len=self._max_seq_len,
        )
        if launch is None:
            return None
        return launch, lut, lut[:num_heads], device

    @staticmethod
    def _ready(families, device: torch.device) -> bool:
        """Whether every family is loaded and the FP8 decode table exists.

        Loads them first outside stream capture, recording a family that
        fails to build or load as unavailable for this process.
        """
        from . import cake_trtllm_mla_blackwell as cake

        if _CAKE_MLA_UNAVAILABLE_FAMILIES.intersection(families):
            return False
        if torch.cuda.is_current_stream_capturing():
            return _CAKE_MLA_LOADED_FAMILIES.issuperset(families) and (
                device in cake._FP8_LUTS
            )
        for family in families - _CAKE_MLA_LOADED_FAMILIES:
            try:
                cake.get_cake_trtllm_mla_blackwell_family(family)
            except Exception as exc:
                _CAKE_MLA_UNAVAILABLE_FAMILIES.add(family)
                logger.warning(
                    "[MLA] Cake family %s unavailable; backend='auto' no longer "
                    "offers it in this process: %s: %s",
                    family,
                    type(exc).__name__,
                    exc,
                )
                return False
            _CAKE_MLA_LOADED_FAMILIES.add(family)
        cake._fp8_lut(device)
        return True

    def supports(self, inputs) -> bool:
        """Whether the route launches for these inputs (within the row cap)."""
        return self._families(inputs, (inputs[0].size(0),)) is not None

    def _serves_bucket(self, inputs, batch_size: int, profile) -> bool:
        """Whether every batch of the tuning bucket of ``batch_size`` launches
        the programs the profiled batch launches, building them.

        The bucket serves the batches up to the next bucket (the max of the
        profile's batch dimension); the largest bucket also serves every larger
        batch, beyond the row cap. The profile measures one batch, so a bucket
        whose batches cross a program boundary (value-split count, one-wave or
        ordered CLC launch) is not offered: its other batches were never
        measured. Building here lets a missing toolchain or architecture (e.g.
        an nvcc without sm_103a) drop Cake from tuning instead of failing the
        tuned call.
        """
        query = inputs[0]
        dim = profile.shapes[0][0] if profile is not None else None
        last = getattr(dim, "max", batch_size + 1) - 1
        if math.isinf(last) or last * query.size(1) > _CAKE_MLA_MAX_ROWS:
            return False
        families = self._families(inputs, range(batch_size, int(last) + 1))
        return bool(families) and self._ready(families, query.device)

    def get_valid_tactics(self, inputs, profile) -> List[Tuple[str, Any]]:
        # Cake is profiled only against a first-stage winner of this profile:
        # alone it would win without a comparison.
        if (
            self._incumbent is None
            or profile is None
            or not self._incumbent.times_profile(profile)
        ):
            return []
        winner = self._incumbent.profile_winner(profile)
        if winner is None:
            return []
        return (
            [winner] if self._serves_bucket(inputs, inputs[0].size(0), profile) else []
        )

    def validate_tactic(self, inputs, tactic) -> bool:
        """Whether a cached Cake choice launches for these inputs now and still
        beats the first stage's current winner for them.

        The tuner calls it on every cache hit; ``False`` makes the hit a miss.
        It records its outcome (``_validation``): a choice that does not launch
        for these sizes or no longer names the first stage's winner is
        ``_CAKE_VALID_STABLE_MISS`` (it holds while the inputs' sizes and the
        tuner's winners are unchanged), a family not loaded or unavailable is
        ``_CAKE_VALID_UNREADY`` (it can change in either direction).
        """
        # Counted as unready until the outcome is known, so a raised error
        # counts too.
        self._validation = _CAKE_VALID_UNREADY
        self._validated_families = None
        self._unready += 1
        if self._incumbent is None or not isinstance(tactic, tuple):
            return self._stable_miss()
        families = self._families(inputs, (inputs[0].size(0),))
        if not families or tactic != self._incumbent.call_winner(inputs):
            return self._stable_miss()
        if not self._ready(families, inputs[0].device):
            return False
        self._unready -= 1
        self._validation = _CAKE_VALID_OK
        self._validated_families = families
        return True

    def _stable_miss(self) -> bool:
        self._unready -= 1
        self._validation = _CAKE_VALID_STABLE_MISS
        return False

    def validate_tactic_for_profile(self, inputs, tactic, profile) -> bool:
        """Validate a bucket reused during tuning against the bucket's batches
        and the first stage's current winner for the bucket."""
        if self._incumbent is None or not isinstance(tactic, tuple):
            return False
        batch_size = profile.get_opt_shapes()[0][0]
        return tactic == self._incumbent.profile_winner(
            profile
        ) and self._serves_bucket(inputs, batch_size, profile)

    def get_cache_key_extras(self, inputs):
        q = inputs[0]
        return (
            q.dtype,
            self._qk_nope_head_dim,
            self._kv_lora_rank,
            self._qk_rope_head_dim,
            self._page_size,
            _round_to_seq_len_bucket(self._max_seq_len),
            self.sinks is not None,
        )

    def forward(self, inputs, tactic: Any = -1, do_preparation: bool = False, **kwargs):
        if do_preparation:
            # get_valid_tactics already built the programs (_ready); the tuner's
            # preparation call must not launch the decode.
            return None
        query, block_tables, seq_lens, out = inputs[:4]
        plan = self._plan
        if plan is not None:
            # The launch the second stage resolved for exactly these inputs
            # (their sizes and contiguity were checked for this call): the
            # kernels and arguments of the dispatcher's stateless mode, without
            # re-deriving them.
            launch, _, placeholder, device = plan
            _cake_mla_module().launch_device_rows(
                launch,
                query=query,
                kv_cache=self.kv_cache,
                out=out,
                seq_lens=seq_lens,
                block_tables=block_tables,
                sink_arg=placeholder,
                device=device,
                stream=torch._C._cuda_getCurrentRawStream(device.index),
                stateless=True,
            )
            return out
        return _cake_mla_module().trtllm_mla_blackwell_decode(
            query=query,
            kv_cache=self.kv_cache,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=self._max_seq_len,
            qk_nope_head_dim=self._qk_nope_head_dim,
            kv_lora_rank=self._kv_lora_rank,
            qk_rope_head_dim=self._qk_rope_head_dim,
            sparse_mla_top_k=0,
            out=out,
            bmm1_scale=self._bmm1_scale,
            bmm2_scale=self._bmm2_scale,
            sinks=self.sinks,
            skip_softmax_threshold_scale_factor=None,
            enable_pdl=self._enable_pdl,
            uses_shared_paged_kv_idx=True,
            lse=None,
            return_lse=False,
            cum_seq_lens_q=None,
            max_q_len=None,
            multi_ctas_kv_counter_buffer=None,
            sparse_mla_top_k_lens=None,
            enable_dcp=False,
            backend="cake",
            stateless=True,
        )

    def __hash__(self):
        # Tensor attributes differ per dispatcher call; get_cache_key_extras
        # carries every tactic-determining value.
        return hash(type(self))


# The class name keys the runner's tuner entries (AutoTuner.may_select_runner_class).
_CAKE_MLA_RUNNER = _CakeTrtllmMlaBlackwellRunner.__name__

# backend="auto" selects in two stages. The first stage is the TRTLLM-GEN /
# CuTe DSL selection under its own op, runners, tuning config and cache keys.
# Where Cake is offered, the second stage, a separate op, compares the
# first-stage winner with Cake per tuning bucket.
_MLA_DECODE_OP = "trtllm_batch_decode_mla"
_MLA_CAKE_MARGIN_OP = "trtllm_batch_decode_mla_cake_margin"

# The second stage's tie, in ms, at the time the first stage measured for its
# winner (I1). The tuner times a single cold replay as the median of ten
# CUDA-event samples; on SM100a and SM103a these samples step in units of
# 2.048 us, so a median lies on a 1.024 us grid and a decode kernel of about
# 10 us measures a real gap of up to one step as either no difference or one
# full step. 1.25 steps (between one and one and a half steps) covers a
# one-step difference plus its jitter. The tuner's tolerance is relative, so a
# bucket is timed under this tie divided by I1, and the tuner applies that
# fraction to its second-stage measurement of the winner (I2): Cake is kept
# unless the winner measures faster by more than this tie times I2 / I1. The
# two measurements of the winner can differ by a timer step or more, so the
# bound is not exactly 1.25 steps. The offer table admits only units where
# Cake measured faster offline, so the second stage only guards against a
# measurable regression.
_MLA_CAKE_STAGE_TWO_TIE_MS = 1.25 * 2.048e-3


def _mla_effective_tuning_config(tuner: AutoTuner, config: TuningConfig):
    """``config`` as ``AutoTuner.choose_one`` applies it: with the active
    ``autotune()`` overrides and measurement policy."""
    with tuner._lock:
        if (
            tuner._override_tuning_buckets is not None
            or tuner._override_round_up
            or tuner._override_cuda_graph_profile_replays is not None
        ):
            config = tuner._apply_tuning_overrides(config)
        policy = tuner._effective_measure_policy
        if policy is not None:
            config = tuner._apply_measure_policy(config, policy)
    return config


class _MlaStageOneWinnerRunner(TunableRunner):
    """The incumbent of the second selection stage: per profile, the runner and
    tactic the first stage selected, whose tactic is ``(runner class name,
    first-stage tactic)``.

    It offers a profile only when the first stage holds a winner for it, looked
    up exactly as the first stage looks it up, and ``cake`` serves the
    profile's bucket: a profile without Cake has nothing to compare, so it is
    not timed. Its cache key covers the first-stage runners and their
    cache-key extras.
    """

    def __init__(
        self,
        runners: Sequence[TunableRunner],
        tuning_config: TuningConfig,
        inputs: List[torch.Tensor],
        cake: Optional["_CakeTrtllmMlaBlackwellRunner"] = None,
    ) -> None:
        self._runners = list(runners)
        self._config = tuning_config
        self._inputs = inputs
        self._cake = cake
        # The batches (profile opt values) the second stage times now; ``None``:
        # every batch. See ``_mla_cake_stage_two_tune``.
        self._timed_batches: Optional[FrozenSet[int]] = None

    def _lookup(self, tuner: AutoTuner, config: TuningConfig, shapes, profile):
        """The first stage's lookup of ``shapes`` (under ``tuner._lock``)."""
        from ..autotuner.autotuner import _ProfilingCandidates

        if tuner.is_tuning_mode:
            # As choose_one's lookups while tuning.
            return tuner.search_cache(
                _MLA_DECODE_OP,
                self._runners,
                shapes,
                config,
                inputs=self._inputs,
                require_profiling_policy=True,
                profiling_candidates=_ProfilingCandidates.from_runners(
                    self._runners, self._inputs
                ),
                validation_profile=profile,
            )
        return tuner.search_cache(
            _MLA_DECODE_OP, self._runners, shapes, config, inputs=self._inputs
        )

    def _winner(self, shapes, profile) -> Optional[Tuple[str, Any]]:
        tuner = AutoTuner.get()
        with tuner._lock:
            config = _mla_effective_tuning_config(tuner, self._config)
            hit, runner_id, tactic, _ = self._lookup(tuner, config, shapes, profile)
        if not hit:
            return None
        return type(self._runners[runner_id]).__name__, tactic

    def profile_winner(self, profile) -> Optional[Tuple[str, Any]]:
        """The first-stage winner of a tuning profile, or ``None``."""
        return self._winner(profile.get_opt_shapes(), profile)

    def profile_time(self, profile) -> Optional[float]:
        """The time (ms) the first stage measured for its winner of a tuning
        profile, from the winner's selection record (of this process's tuning
        or of a loaded cache file), or ``None`` without a winner or record."""
        tuner = AutoTuner.get()
        shapes = profile.get_opt_shapes()
        with tuner._lock:
            config = _mla_effective_tuning_config(tuner, self._config)
            hit, runner_id, _, _ = self._lookup(tuner, config, shapes, profile)
            if not hit:
                return None
            runner = self._runners[runner_id]
            key = tuner._get_cache_key(
                _MLA_DECODE_OP,
                runner,
                shapes,
                config,
                runner.get_cache_key_extras(self._inputs),
            )
            identity = tuner._winner_cache_identity()
            record = tuner._profiling_cache_selections.get((identity, key))
            if record is None and identity is None:
                record = tuner._file_config_selections.get(key.file_key)
        time_ms = record.get("time_ms") if isinstance(record, dict) else None
        if not isinstance(time_ms, (int, float)) or not 0 < time_ms < math.inf:
            return None
        return float(time_ms)

    def times_profile(self, profile) -> bool:
        """Whether the second stage times this profile now."""
        timed = self._timed_batches
        return timed is None or profile.get_opt_shapes()[0][0] in timed

    def call_winner(self, inputs) -> Optional[Tuple[str, Any]]:
        """The first-stage winner for these call inputs, or ``None``."""
        return self._winner(AutoTuner.get()._get_input_sizes(inputs), None)

    def get_valid_tactics(self, inputs, profile) -> List[Tuple[str, Any]]:
        if not self.times_profile(profile):
            return []
        winner = self.profile_winner(profile)
        if winner is None or (
            self._cake is not None
            and not self._cake._serves_bucket(inputs, inputs[0].size(0), profile)
        ):
            return []
        return [winner]

    def validate_tactic_for_profile(self, inputs, tactic, profile) -> bool:
        """A reused bucket stays valid while the first stage keeps its winner.
        The default tactic ``-1`` (raced by ``autotune_v2`` contexts) beat or
        tied Cake as well, so it stays valid."""
        return tactic == -1 or tactic == self.profile_winner(profile)

    def get_cache_key_extras(self, inputs):
        return tuple(
            (type(runner).__name__, runner.get_cache_key_extras(inputs))
            for runner in self._runners
        )

    def forward(self, inputs, tactic: Any = -1, do_preparation: bool = False, **kwargs):
        for runner in self._runners:
            if isinstance(tactic, tuple) and type(runner).__name__ == tactic[0]:
                return runner(
                    inputs, tactic=tactic[1], do_preparation=do_preparation, **kwargs
                )
        return self._runners[0](
            inputs, tactic=-1, do_preparation=do_preparation, **kwargs
        )

    def __hash__(self):
        # Cache-key extras carry the first-stage runners' identity.
        return hash(type(self))


@functools.lru_cache(maxsize=256)
def _mla_cake_margin_tuning_config(
    config: TuningConfig, batch_ranges: Tuple[Tuple[int, int], ...]
) -> Optional[TuningConfig]:
    """The second stage's tuning config: the first stage's, generating only the
    buckets within ``batch_ranges`` and the bucket after each of them, which
    bounds its profile's batches (no candidate offers a tactic there, so it
    is not timed). Its batch mapping, and so every cache key, is the first
    stage's. ``None`` when no bucket is within the ranges.

    ``autotune(tuning_buckets=...)`` replaces the generated buckets with its
    own; the candidates then still offer tactics only in the offered buckets,
    so only those are timed."""
    spec, *others = config.dynamic_tensor_specs
    if callable(spec.gen_tuning_buckets):
        # Generated per call: the second stage generates the first stage's.
        return config
    buckets = tuple(sorted(set(spec.gen_tuning_buckets)))
    kept: Set[int] = set()
    for index, bucket in enumerate(buckets):
        if any(first <= bucket <= last for first, last in batch_ranges):
            kept.update(buckets[index : index + 2])
    if not kept:
        return None
    return replace(
        config,
        dynamic_tensor_specs=(
            replace(spec, gen_tuning_buckets=tuple(sorted(kept))),
            *others,
        ),
    )


# Second-stage decisions of lookup-only calls, keyed by
# ``(memo_key, batch ranges, first-stage selection)``, where ``memo_key`` (the
# caller's) holds every argument and tensor size, dtype and device the two
# stages' lookups and the Cake checks read. A value is ``(weak reference
# to the tuner, tuner state, offer table, families, launch plan, FP8 decode
# table, copy of the split-KV device ranges)``; ``families`` is ``None`` for a
# decision to run the first stage's selection. The offer table and the Cake
# module's ``_SPLITKV_RANGE`` decide the launched domain with the memo key (its
# device and max_seq_len), so an entry is reused only while the offer table is
# the object and the ranges equal the values it was decided with (also after an
# in-place change of a range). The weak reference keeps an entry from holding a
# discarded tuner alive. See ``_mla_cake_stage_two``. Bounded like
# ``_CAKE_MLA_TABLE_MEMO``.
# The dispatcher memoizes whole decisions in ``_CAKE_MLA_AUTO_MEMO`` instead;
# this memo serves callers of ``_mla_cake_stage_two`` without one.
_CAKE_MLA_STAGE_TWO_MEMO: Dict[Any, Any] = {}


def _mla_cake_memo_state(tuner: AutoTuner) -> Optional[tuple]:
    """What, besides the call's arguments, decides a lookup of the two stages
    and whether Cake joins now: the generation of the tuner's winner sources
    (advanced by every change of the in-memory winners and loaded configs of
    every tuner), the active ``autotune()`` overrides, and the settings
    ``_mla_cake_memo_settings`` reads (whether bundled configs are read and
    whether the Cake candidates are switched on). ``None`` when a lookup reads
    sources the generation does not cover, is recorded or is skipped: while
    tuning, with a managed ``autotune_v2`` store, inside an ``autotune_v2``
    context, under replay diagnostics, with a measurement policy, or while the
    thread's ``skip_ops`` names either stage's op.

    With a state, ``cake_auto_candidate_open`` is a function of it: not
    tuning, so no process group tunes; no managed store, so
    ``AutoTuner.may_select_runner_class`` reads only the tuner's winner
    sources (covered by the generation) and, with
    ``FLASHINFER_AUTOTUNER_LOAD_FROM_FILE=1``, the bundled configs; and the
    switch ``FLASHINFER_DISABLE_CAKE_AUTO_CANDIDATES``."""
    if tuner.is_tuning_mode or tuner._managed_cache is not None:
        return None
    # The thread's skip_ops and override stacks, read as
    # ``AutoTuner._effective_skip_ops`` and ``_get_override_stack`` read them,
    # without creating them (an absent stack is an empty one).
    skip_stack = getattr(tuner._skip_ops_local, "stack", None)
    if skip_stack:
        skip_ops = skip_stack[-1]
        if _MLA_DECODE_OP in skip_ops or _MLA_CAKE_MARGIN_OP in skip_ops:
            return None
    local = tuner._v2_local
    if (
        local.active
        or local.replay_report is not None
        or local.strict_replay
        or local.measure is not None
    ):
        return None
    stack = getattr(tuner._override_local, "stack", None)
    environ = os.environ
    return (
        _autotuner_module._winner_sources_generation,
        tuple(stack[-1][:3]) if stack else None,
        # _mla_cake_memo_settings(), inlined.
        (
            environ.get(_AUTOTUNER_LOAD_FROM_FILE_ENV, "0"),
            environ.get(_CAKE_AUTO_CANDIDATES_SWITCH_ENV, "0") != "1",
        ),
    )


# The environment variables of ``_mla_cake_memo_settings``. The second is the
# switch ``cake_auto_candidates_enabled`` reads, compared as it compares it.
_AUTOTUNER_LOAD_FROM_FILE_ENV = "FLASHINFER_AUTOTUNER_LOAD_FROM_FILE"
_CAKE_AUTO_CANDIDATES_SWITCH_ENV = _CAKE_AUTO_CANDIDATES_ENV_VAR


def _mla_cake_memo_settings() -> Tuple[str, bool]:
    """The environment settings in ``_mla_cake_memo_state``, read on every
    call: ``FLASHINFER_AUTOTUNER_LOAD_FROM_FILE`` and whether the Cake
    candidates are switched on (the read ``cake_auto_candidate_open`` makes,
    as ``cake_auto_candidates_enabled``)."""
    environ = os.environ
    return (
        environ.get(_AUTOTUNER_LOAD_FROM_FILE_ENV, "0"),
        environ.get(_CAKE_AUTO_CANDIDATES_SWITCH_ENV, "0") != "1",
    )


# The last result of the Cake candidate gate under a memo state: ``(weak
# reference to the tuner, state, open)``, replaced as one tuple. With a state
# the gate is a function of the tuner and the state (see
# ``_mla_cake_memo_state``), so a call that misses ``_CAKE_MLA_AUTO_MEMO``
# (a new key, such as a ``max_seq_len`` of another bucket) under the last
# gate's state pays no gate.
_CAKE_MLA_GATE_MEMO: List[Any] = [None]


def _mla_cake_candidate_open(tuner: AutoTuner, state: tuple) -> bool:
    """``cake_auto_candidate_open`` of the Cake MLA runner for ``tuner`` under
    ``state`` (the call's ``_mla_cake_memo_state``, not ``None``), memoized for
    the last tuner and state. Stored only while ``tuner`` is the tuner the
    gate reads and the settings of ``state`` are still the environment's."""
    last = _CAKE_MLA_GATE_MEMO[0]
    if last is not None and last[0]() is tuner and last[1] == state:
        return last[2]
    candidate_open = cake_auto_candidate_open(_CAKE_MLA_RUNNER)
    if AutoTuner.get() is tuner and state[2] == _mla_cake_memo_settings():
        _CAKE_MLA_GATE_MEMO[0] = (weakref.ref(tuner), state, candidate_open)
    return candidate_open


def _cake_mla_families_ready(families: frozenset, device: torch.device, lut) -> bool:
    """Whether every family is loaded and none unavailable, and ``lut`` is the
    device's FP8 decode table: ``_ready`` without building anything."""
    unavailable = _CAKE_MLA_UNAVAILABLE_FAMILIES
    if unavailable and not unavailable.isdisjoint(families):
        return False
    return _CAKE_MLA_LOADED_FAMILIES.issuperset(families) and (
        _cake_mla_module()._FP8_LUTS.get(device) is lut
    )


def _mla_cake_stage_two_runners(
    tuner: AutoTuner,
    incumbent: _MlaStageOneWinnerRunner,
    cake: _CakeTrtllmMlaBlackwellRunner,
) -> List[TunableRunner]:
    """The second stage's candidates in enumeration order: Cake first, so an
    exact tie with the first stage's winner selects Cake (the tuner keeps the
    first of equal times), and the tuner's tie tolerance, which keeps an
    earlier candidate within it, keeps Cake.

    Only offered units reach the second stage, and the offer table admits a
    unit only where Cake measured a median of at most 0.98 and every round at
    most 1.00 of the best other candidate. The tuner's single cold replay
    quantizes a microsecond decode kernel to steps of about 2 us, so a gap of
    up to one step in an offered unit is the timer's resolution, not evidence
    that the first stage's winner is faster. ``_mla_cake_stage_two_tune``
    therefore times each bucket under the relative tolerance that is
    ``_MLA_CAKE_STAGE_TWO_TIE_MS`` at the first stage's time of its winner (or
    under ``autotune(tie_tolerance=...)`` when that is wider): the tuner
    re-times Cake and keeps it while its mean stays within the tolerance of
    its second-stage measurement of the winner, and a first-stage winner
    faster by more still wins.

    Inside an ``autotune_v2`` context the tuner also races the default tactic
    ``-1`` of the first candidate, even where that candidate offers no tactic
    for a bucket. Cake does not serve every bucket, so there the first-stage
    winner stays first (as before this order existed, an exact tie keeps it);
    its ``-1`` runs the first stage's default runner. The order decides no
    cache key, and every consumer identifies Cake by the runner object, never
    by its position."""
    if tuner._in_v2_context:
        return [incumbent, cake]
    return [cake, incumbent]


def _mla_cake_stage_two_tune(
    tuner: AutoTuner,
    pair: List[TunableRunner],
    incumbent: _MlaStageOneWinnerRunner,
    config: TuningConfig,
    inputs: List[torch.Tensor],
    batch_ranges: Tuple[Tuple[int, int], ...],
) -> Tuple[TunableRunner, Any]:
    """Tunes the second stage (``choose_one`` of ``_MLA_CAKE_MARGIN_OP`` over
    ``pair``, ``_mla_cake_stage_two_runners``) and returns its selection for
    ``inputs``.

    Outside ``autotune_v2`` contexts the tuner's ``tie_tolerance`` is
    relative and holds for a whole ``choose_one``, so each bucket within
    ``batch_ranges`` is timed under the tolerance ``T / I1`` (at most 0.5),
    where ``T`` is ``_MLA_CAKE_STAGE_TWO_TIE_MS`` and ``I1`` the time the
    first stage measured for its winner there
    (``_MlaStageOneWinnerRunner.profile_time``), or under
    ``autotune(tie_tolerance=...)`` when that is larger or there is no such
    time. The tuner applies the tolerance to its own measurement of the
    first-stage winner, ``I2``, so Cake is kept in a bucket unless that winner
    measures faster by more than ``T * I2 / I1``: ``T`` where both stages
    measure the winner alike. The buckets of one tolerance are timed in one ``choose_one``, in
    which the other buckets offer no tactic. The buckets, their profiles and
    every cache key are the ones a single ``choose_one`` uses; with one
    tolerance for every bucket it is that single ``choose_one``. Inside an
    ``autotune_v2`` context (where the first-stage winner is the first
    candidate) the tuner's own tolerance decides, as for other ops."""
    if tuner._in_v2_context:
        return tuner.choose_one(_MLA_CAKE_MARGIN_OP, pair, config, inputs)
    # choose_one tunes under the tuner's lock (an RLock); holding it across
    # the passes keeps another thread from changing the winners between them.
    with tuner._lock:
        default = tuner._tie_tolerance
        groups: Dict[float, Set[int]] = {}
        for profile in tuner._generate_optimization_profiles(
            _mla_effective_tuning_config(tuner, config), inputs
        ):
            batch = profile.get_opt_shapes()[0][0]
            if not _in_batch_ranges(batch, batch_ranges):
                continue  # Cake never serves it, so it is never timed.
            time_ms = incumbent.profile_time(profile)
            tolerance = default
            if time_ms is not None:
                # autotune() accepts tolerances below 1.
                tie = min(_MLA_CAKE_STAGE_TWO_TIE_MS / time_ms, 0.5)
                tolerance = max(default, tie)
            groups.setdefault(tolerance, set()).add(batch)
        tolerances: List[Tuple[float, Optional[FrozenSet[int]]]]
        if len(groups) <= 1:
            tolerances = [(next(iter(groups), default), None)]
        else:
            tolerances = [(tol, frozenset(batches)) for tol, batches in groups.items()]
        stack = tuner._get_override_stack()
        top = stack[-1] if stack else (None, False, None, None)
        selection: Tuple[TunableRunner, Any] = (pair[0], -1)
        try:
            for tolerance, batches in tolerances:
                incumbent._timed_batches = batches
                pushed = tolerance != default
                if pushed:
                    stack.append((*top[:3], tolerance, *top[4:]))
                try:
                    # Every choose_one ends with the same lookup of the call's
                    # bucket, so the last one returns the call's selection.
                    selection = tuner.choose_one(
                        _MLA_CAKE_MARGIN_OP, pair, config, inputs
                    )
                finally:
                    if pushed:
                        stack.pop()
        finally:
            incumbent._timed_batches = None
        return selection


def _mla_cake_stage_two(
    tuner: AutoTuner,
    runners: List[TunableRunner],
    tuning_config: TuningConfig,
    inputs: List[torch.Tensor],
    selected: Tuple[TunableRunner, Any],
    cake: Union[
        _CakeTrtllmMlaBlackwellRunner, Callable[[], _CakeTrtllmMlaBlackwellRunner]
    ],
    batch_ranges: Tuple[Tuple[int, int], ...],
    memo_key: Optional[tuple] = None,
    memoizable: Optional[List[Optional[tuple]]] = None,
    memo_state: Optional[tuple] = None,
) -> Tuple[TunableRunner, Any]:
    """The second selection stage: Cake when it beat or tied the first stage's
    winner for this call's bucket (``_mla_cake_stage_two_runners``) and
    launches now, else ``selected``, the first stage's runner and tactic (also
    on every Cake miss, and when tuning the second stage fails).

    While tuning outside stream capture it times the buckets within
    ``batch_ranges`` (``_mla_cake_stage_two_tune``); otherwise, and during
    capture, it only looks up. The
    first stage's lookup records the call in ``autotune_v2`` replay
    diagnostics; the second stage's lookups are not recorded, and a miss
    under strict replay runs the first stage's winner instead of raising.

    ``cake`` is the Cake runner of the call, or a function building it (only
    when the call needs it). With a ``memo_key`` (the dispatcher's), a
    lookup-only decision is memoized per ``(memo_key, batch_ranges, first-stage
    selection)`` and reused while ``_mla_cake_memo_state``, the offer table
    and the values of the split-KV device ranges are unchanged: a decision to
    run Cake while its families stay loaded and available (else the call
    looks up again), a decision to run the first stage's selection only when
    no readiness check decided it (a Cake choice that does not launch for
    these sizes, does not name the first stage's winner, or no Cake choice).
    The memo is skipped whenever ``_mla_cake_memo_state`` is ``None``.

    The dispatcher memoizes whole decisions instead (``_CAKE_MLA_AUTO_MEMO``):
    it passes no ``memo_key``, ``memoizable`` (an empty list) and
    ``memo_state``, its ``_mla_cake_memo_state`` read before the first
    stage's lookup, not ``None``. This memo is then neither read nor written,
    and ``memoizable``
    receives the decision of a lookup-only call that this memo would store
    (and of one no second-stage bucket covers): ``None`` for running
    ``selected``, else ``(Cake tactic, families, launch plan, FP8 decode
    table)``; the dispatcher keys the decision on the split-KV device ranges
    as this memo does.
    """
    skip_ops = tuner._effective_skip_ops
    if skip_ops and (_MLA_DECODE_OP in skip_ops or _MLA_CAKE_MARGIN_OP in skip_ops):
        return selected
    lookup_only = not tuner.is_tuning_mode or torch.cuda.is_current_stream_capturing()
    if lookup_only and not _in_batch_ranges(inputs[0].size(0), batch_ranges):
        return selected
    # A Cake choice runs only in place of the first-stage selection it beat or
    # tied (also where replay diagnostics made the first stage's lookup miss).
    beaten = (type(selected[0]).__name__, selected[1])
    state = memo_entry_key = None
    if lookup_only and memoizable is not None:
        state = memo_state
    elif lookup_only and memo_key is not None:
        state = _mla_cake_memo_state(tuner)
    if state is not None and memoizable is None:
        memo_entry_key = (memo_key, batch_ranges, beaten)
        try:
            entry = _CAKE_MLA_STAGE_TWO_MEMO.get(memo_entry_key)
        except TypeError:  # an unhashable first-stage tactic
            entry, state = None, None
        if (
            entry is not None
            and entry[0]() is tuner
            and entry[1] == state
            and entry[2] is _CAKE_MLA_AUTO_OFFERS
            and entry[6] == _cake_mla_module()._SPLITKV_RANGE
        ):
            families = entry[3]
            if families is None:
                return selected
            if _cake_mla_families_ready(families, inputs[0].device, entry[5]):
                if not isinstance(cake, _CakeTrtllmMlaBlackwellRunner):
                    cake = cake()
                cake._plan = entry[4]
                return cake, beaten
    if not isinstance(cake, _CakeTrtllmMlaBlackwellRunner):
        cake = cake()
    # autotune(round_up=True) rebuilds the batch mapping from the profiled
    # buckets, so the second stage then generates the first stage's buckets.
    config = (
        tuning_config
        if tuner._override_round_up
        else _mla_cake_margin_tuning_config(tuning_config, batch_ranges)
    )
    if config is None:
        # A function of the first stage's config, the ranges and the overrides.
        if state is not None and memoizable is not None:
            memoizable.append(None)
        return selected
    incumbent = _MlaStageOneWinnerRunner(runners, tuning_config, inputs, cake)
    cake._incumbent = incumbent
    pair = _mla_cake_stage_two_runners(tuner, incumbent, cake)
    if not lookup_only:
        from ..autotuner.autotuner import _StickyCudaError, _TuningGroupError

        try:
            runner, tactic = _mla_cake_stage_two_tune(
                tuner, pair, incumbent, config, inputs, batch_ranges
            )
        except (_StickyCudaError, _TuningGroupError):
            raise
        except Exception as exc:
            logger.warning(
                "[MLA] tuning the Cake candidate failed; the call runs the "
                "first-stage selection: %s: %s",
                type(exc).__name__,
                exc,
            )
            return selected
        if runner is not cake or tactic != beaten:
            return selected
        try:
            cake_ok = cake.validate_tactic(inputs, tactic)
        except Exception:
            cake_ok = False
        return (cake, tactic) if cake_ok else selected
    # Lookup only, without choose_one's untuned-shape warning: a miss runs the
    # tuned first-stage winner, not the default.
    local = tuner._v2_local
    saved = local.replay_report, local.strict_replay
    local.replay_report, local.strict_replay = None, False
    cake._validation, cake._unready = None, 0
    looked_up = False
    try:
        hit, runner_id, tactic, _ = tuner.search_cache(
            _MLA_CAKE_MARGIN_OP,
            pair,
            tuner._get_input_sizes(inputs),
            _mla_effective_tuning_config(tuner, config),
            inputs=inputs,
            # Strict replay still requires the requested measurement policy.
            require_profiling_policy=saved[1],
        )
        cake_ok = hit and pair[runner_id] is cake and tactic == beaten
        # search_cache returns a hit only after the runner's validate_tactic
        # accepted it for these inputs (AutoTuner._tactic_still_valid on every
        # source), so a Cake hit is validated once; a lookup that returned one
        # without validating it is validated here.
        if cake_ok and cake._validation is None:
            cake_ok = cake.validate_tactic(inputs, tactic)
        looked_up = True
    except Exception:
        cake_ok = False
    finally:
        local.replay_report, local.strict_replay = saved
    plan = None
    if cake_ok:
        try:
            plan = cake._launch_plan(inputs)
        except (RuntimeError, ValueError, KeyError):
            # No resolved launch for these sizes: ``forward`` takes the
            # dispatcher. Any other error is a defect and propagates.
            plan = None
        cake._plan = plan
    # Memoize only a decision no readiness check (or failure) took part in.
    if state is not None and looked_up and cake._unready == 0:
        families = cake._validated_families if cake_ok else None
        lut = _cake_mla_module()._FP8_LUTS.get(inputs[0].device) if cake_ok else None
        if memoizable is not None:
            memoizable.append((beaten, families, plan, lut) if cake_ok else None)
        else:
            if len(_CAKE_MLA_STAGE_TWO_MEMO) > _CAKE_MLA_MEMO_LIMIT:
                _CAKE_MLA_STAGE_TWO_MEMO.clear()
            _CAKE_MLA_STAGE_TWO_MEMO[memo_entry_key] = (
                weakref.ref(tuner),
                state,
                _CAKE_MLA_AUTO_OFFERS,
                families,
                plan,
                lut,
                dict(_cake_mla_module()._SPLITKV_RANGE),
            )
    return (cake, tactic) if cake_ok else selected


# Whole ``backend="auto"`` decisions (both stages) of lookup-only calls of
# listed batches, keyed by the dispatcher's ``memo_key``. A value is ``(weak
# reference to the tuner, tuner state, offer table, decision, copy of the
# split-KV device ranges)``, reused while the ranges equal the copy (as in
# ``_CAKE_MLA_STAGE_TWO_MEMO``); ``decision``
# is ``(index of the first stage's runner in the call's runner names,
# first-stage tactic, cake)``, ``cake`` as ``_mla_cake_stage_two`` reports it
# (``None``: run the first stage's selection; also stored for a call the
# eligibility check rejects for a reason the key decides, and for a call the
# Cake candidate gate ``cake_auto_candidate_open`` left out). The dispatcher
# looks an entry up before that gate: with a state the gate is a function of
# it (see ``_mla_cake_memo_state``), so an entry made under the same state
# had the gate of this call. Bounded like ``_CAKE_MLA_TABLE_MEMO``. A decision
# is exact for the tuner state and offer table read during the call; a
# concurrent change of the offer table or of a setting of the state
# (``FLASHINFER_AUTOTUNER_LOAD_FROM_FILE``,
# ``FLASHINFER_DISABLE_CAKE_AUTO_CANDIDATES``) that is reverted within one
# call (another thread) is not detected.
#
# What the two stages' lookups read besides ``_mla_cake_memo_state``: cache
# keys of the runner class, a per-class hash and the cache-key extras, and a
# tuning config, all functions of ``memo_key`` fields (of ``max_seq_len`` the
# key holds only ``_mla_auto_memo_seq_len``: the extras read its bucket or its
# KV tile count, the tuning config its KV tile count, for the CuTe DSL batch
# cap, and its exact value only for the profiling initializers, which a
# lookup does not run; the Cake domain reads whether it reaches the device's
# split-KV range); and the stored-tactic
# revalidation (``AutoTuner._tactic_still_valid``), which calls no validator
# for TRTLLM-GEN, modular CuTe DSL or the incumbent, accepts only tactic -1 for
# functional monolithic CuTe DSL, and for Cake reads the sizes, the contiguity
# of query, block_tables, seq_lens and out, the first stage's lookup and the
# readiness of the families. No candidate reads tensor data, seq_lens values,
# data pointers, alignment, strides beyond contiguity or workspace contents
# to select. A reused Cake decision therefore re-checks the contiguity the
# eligibility check requires (else the call runs the first stage's selection,
# as that check makes it) and the families' readiness (else it selects).
# The launch plan of a Cake decision reads ``max_seq_len`` only through the
# key's field (its domain through the split-KV side, its split counts through
# the KV tile count), so it is the plan of every ``max_seq_len`` of the key; a
# reused decision builds the selected runner with the call's exact
# ``max_seq_len``, which the kernels of the first stage (and a Cake decision
# without a plan) read.
_CAKE_MLA_AUTO_MEMO: Dict[Any, Any] = {}


def _mla_auto_memo_seq_len(
    max_seq_len: int, device_index: int
) -> Tuple[int, int, Optional[bool]]:
    """What a ``backend="auto"`` decision on the device of ``device_index``
    (one with Cake programs) reads of ``max_seq_len``, the field of the
    dispatcher's memo key that stands for it: its power-of-two bucket (the
    offer table and the TRTLLM-GEN, modular CuTe DSL and Cake cache keys), its
    128-token KV tile count (the monolithic CuTe DSL cache key and batch cap,
    and the split counts of a resolved Cake launch) and whether it reaches the
    device's split-KV range (the Cake domain; ``None`` for a device
    ``_SPLITKV_RANGE`` does not list). The comparison is kept apart from the
    bucket: a range need not start at a bucket edge. Calls whose
    ``max_seq_len`` gives the same value select the same runner and tactic and
    resolve the same Cake launch, so a caller that varies ``max_seq_len``
    selects once per bucket, KV tile count and side of the range."""
    cake = _cake_mla_module()
    splitkv_range = cake._SPLITKV_RANGE.get(cake._device_facts(device_index))
    return (
        _round_to_seq_len_bucket(max_seq_len),
        (max_seq_len + 127) // 128,
        None if splitkv_range is None else max_seq_len >= splitkv_range[0],
    )


def _mla_cake_auto_memo_get(
    tuner: AutoTuner,
    memo_key: tuple,
    state: tuple,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    out: torch.Tensor,
) -> Optional[tuple]:
    """The memoized decision of this call, ``(first-stage runner index,
    first-stage tactic, cake)``, or ``None`` when the call selects: no entry,
    another tuner, offer table, split-KV device ranges or ``state`` (the call's
    ``_mla_cake_memo_state``, not ``None``), or a Cake decision whose families
    are not ready. A Cake decision for tensors that are not all contiguous
    runs the first stage's selection (``cake`` ``None``). Raises ``TypeError``
    for an unhashable key."""
    entry = _CAKE_MLA_AUTO_MEMO.get(memo_key)
    if (
        entry is None
        or entry[0]() is not tuner
        or entry[2] is not _CAKE_MLA_AUTO_OFFERS
        or entry[1] != state
        or entry[4] != _cake_mla_module()._SPLITKV_RANGE
    ):
        return None
    decision = entry[3]
    cake = decision[2]
    if cake is None:
        return decision
    if not (
        query.is_contiguous()
        and kv_cache.is_contiguous()
        and block_tables.is_contiguous()
        and seq_lens.is_contiguous()
        and out.is_contiguous()
    ):
        return decision[0], decision[1], None
    if not _cake_mla_families_ready(cake[1], query.device, cake[3]):
        return None
    return decision


def _mla_cake_auto_memo_put(
    tuner: AutoTuner,
    memo_key: tuple,
    state: tuple,
    offers: Any,
    runners: List[TunableRunner],
    selected: Tuple[TunableRunner, Any],
    cake: Optional[tuple],
) -> None:
    """Memoize the decision of a lookup-only call: ``selected`` is the first
    stage's ``(runner, tactic)``, ``cake`` the second stage's decision as
    ``_mla_cake_stage_two`` reports it (``None`` also when the Cake candidate
    was not open), both made under ``state``, read before the Cake candidate
    gate and the first stage's lookup, and the offer table ``offers``, read
    before the batch ranges. Only while ``offers`` is the offer table and the
    settings of ``state`` are still the environment's (the gate and the
    lookups read them again), and when ``selected`` is one of ``runners``."""
    if offers is not _CAKE_MLA_AUTO_OFFERS or state[2] != _mla_cake_memo_settings():
        return
    index = None
    for position, runner in enumerate(runners):
        if runner is selected[0]:
            index = position
            break
    if index is None:
        return
    if len(_CAKE_MLA_AUTO_MEMO) > _CAKE_MLA_MEMO_LIMIT:
        _CAKE_MLA_AUTO_MEMO.clear()
    _CAKE_MLA_AUTO_MEMO[memo_key] = (
        weakref.ref(tuner),
        state,
        offers,
        (index, selected[1], cake),
        dict(_cake_mla_module()._SPLITKV_RANGE),
    )


def _validate_mla_dcp_args(
    *,
    query: torch.Tensor,
    backend: str,
    sinks: Optional[List[torch.Tensor]],
    cum_seq_lens_q: Optional[torch.Tensor],
    max_q_len: Optional[int],
    return_lse: bool,
    enable_dcp: bool,
    cp_world: int,
    cp_rank: int,
    causal_seqlens_kv_global: Optional[torch.Tensor],
) -> str:
    """Validate the public DCP contract and return the effective backend."""
    if not isinstance(enable_dcp, bool):
        raise TypeError(f"enable_dcp must be a bool, got {type(enable_dcp).__name__}")
    if not isinstance(cp_world, int) or isinstance(cp_world, bool) or cp_world <= 0:
        raise ValueError(f"cp_world must be a positive integer, got {cp_world!r}")
    if not isinstance(cp_rank, int) or isinstance(cp_rank, bool):
        raise TypeError(f"cp_rank must be an integer, got {type(cp_rank).__name__}")

    if not enable_dcp:
        nondefault = []
        if cp_world != 1:
            nondefault.append(f"cp_world={cp_world}")
        if cp_rank != 0:
            nondefault.append(f"cp_rank={cp_rank}")
        if causal_seqlens_kv_global is not None:
            nondefault.append("causal_seqlens_kv_global")
        if nondefault:
            raise ValueError(
                "DCP arguments require enable_dcp=True; got " + ", ".join(nondefault)
            )
        return backend

    is_var_q = cum_seq_lens_q is not None
    if is_var_q:
        if not isinstance(cum_seq_lens_q, torch.Tensor):
            raise TypeError(
                "cum_seq_lens_q must be a torch.Tensor when DCP is enabled, got "
                f"{type(cum_seq_lens_q).__name__}"
            )
        if query.ndim != 3:
            raise ValueError(
                "DCP with cum_seq_lens_q requires a compact query with shape "
                "[total_q, num_heads, head_dim_qk]"
            )
        batch_size = cum_seq_lens_q.numel() - 1
    else:
        if query.ndim != 4:
            raise ValueError(
                "DCP without cum_seq_lens_q requires a dense query with shape "
                "[batch_size, q_len_per_request, num_heads, head_dim_qk]"
            )
        batch_size = query.shape[0]
    if not 0 <= cp_rank < cp_world:
        raise ValueError(
            f"cp_rank must satisfy 0 <= cp_rank < cp_world, got "
            f"cp_rank={cp_rank}, cp_world={cp_world}"
        )
    if backend not in ("auto", "cute-dsl", "cake"):
        raise ValueError(
            f"enable_dcp=True is only supported by backend='cute-dsl' and the "
            f"backend='cake' Kimi-K3 FP8 route, got backend={backend!r}"
        )
    if not return_lse:
        raise ValueError(
            "enable_dcp=True requires return_lse=True so rank-local "
            "attention states can be merged"
        )
    if sinks is not None:
        raise ValueError(
            "DCP cannot be combined with sinks: DCP requires monolithic "
            "CuTeDSL MLA, while sinks require the modular implementation"
        )
    if causal_seqlens_kv_global is None:
        raise ValueError("causal_seqlens_kv_global is required when enable_dcp=True")
    if not isinstance(causal_seqlens_kv_global, torch.Tensor):
        raise TypeError(
            "causal_seqlens_kv_global must be a torch.Tensor, got "
            f"{type(causal_seqlens_kv_global).__name__}"
        )
    if causal_seqlens_kv_global.dtype != torch.int32:
        raise ValueError(
            "causal_seqlens_kv_global must have dtype torch.int32, got "
            f"{causal_seqlens_kv_global.dtype}"
        )
    if not causal_seqlens_kv_global.is_cuda:
        raise ValueError("causal_seqlens_kv_global must be a CUDA tensor")
    if causal_seqlens_kv_global.device != query.device:
        raise ValueError(
            "causal_seqlens_kv_global must be on the query device "
            f"{query.device}, got {causal_seqlens_kv_global.device}"
        )
    if tuple(causal_seqlens_kv_global.shape) != (batch_size,):
        raise ValueError(
            "causal_seqlens_kv_global must have shape "
            f"({batch_size},), got {tuple(causal_seqlens_kv_global.shape)}"
        )
    if not causal_seqlens_kv_global.is_contiguous():
        raise ValueError("causal_seqlens_kv_global must be contiguous")

    # DCP masking exists in the monolithic CuTeDSL implementation and in the Cake Kimi-K3 FP8
    # route; backend="auto" keeps CuTeDSL.
    return "cake" if backend == "cake" else "cute-dsl"


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
    use_fp16_softmax: Optional[bool] = None,
    cute_dsl_impl: str = "auto",
    cum_seq_lens_q: Optional[torch.Tensor] = None,
    max_q_len: Optional[int] = None,
    enable_dcp: bool = False,
    cp_world: int = 1,
) -> Optional[str]:
    """Return None if cute-dsl can handle this call, else a human-readable reason.

    Used by both the explicit ``backend="cute-dsl"`` path (raises the reason
    as a ``ValueError``) and the ``backend="auto"`` filter (silently drops
    cute-dsl from the runners list).
    """
    cc = get_compute_capability(query.device)
    if cc[0] < 10:
        return f"cute-dsl backend (MLA decode kernel) requires SM100+, got SM{cc[0]}{cc[1]}"
    from ..cute_dsl.availability import is_cute_dsl_arch_supported

    if not is_cute_dsl_arch_supported(*cc):
        return (
            "cute-dsl backend (MLA decode kernel): the installed CuTe DSL "
            f"does not support sm_{cc[0]}{cc[1]}"
        )
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
    if use_fp16_softmax:
        return "cute-dsl backend (MLA decode kernel) does not support use_fp16_softmax"
    # LSE is supported on the monolithic path; the modular path raises a
    # clear NotImplementedError in wrappers/batch_mla.py if it gets picked
    # for an LSE request (e.g. when ``sinks`` forces the modular dispatch).
    # We don't pre-reject here so that the common case
    # (cute_dsl_impl="auto" + LSE + no sinks → monolithic) goes through.

    is_var_q = cum_seq_lens_q is not None
    if is_var_q:
        if query.ndim != 3 or max_q_len is None:
            return (
                "cute-dsl backend (MLA decode kernel) requires compact "
                "query [total_q, H, D] and max_q_len for variable Q"
            )
        q_len = max_q_len
        num_heads = query.shape[1]
    else:
        if query.ndim != 4:
            return (
                "cute-dsl backend (MLA decode kernel) requires query rank 4 for fixed Q"
            )
        _, q_len, num_heads, _ = query.shape
    try:
        from ..cute_dsl.attention.mla_dispatch import _resolve_impl

        resolved_impl = _resolve_impl(
            requested=cute_dsl_impl,
            kwargs={
                "sinks": sinks,
                "cum_seq_lens_q": cum_seq_lens_q,
                "max_q_len": max_q_len,
                "enable_dcp": enable_dcp,
            },
        )
    except (TypeError, ValueError, ImportError) as e:
        return f"cute-dsl backend (MLA decode kernel): {e}"

    try:
        if resolved_impl == "monolithic":
            from ..cute_dsl.attention.monolithic.mla_decode import (
                _check_can_implement as check_monolithic_can_implement,
            )
        else:
            from ..cute_dsl.attention.wrappers.batch_mla import (
                _check_can_implement as check_modular_can_implement,
            )

        check_kwargs = dict(
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
        if resolved_impl == "monolithic":
            check_kwargs.update(enable_dcp=enable_dcp, cp_world=cp_world)
            check_monolithic_can_implement(**check_kwargs)
        else:
            check_modular_can_implement(**check_kwargs)
    except (TypeError, ValueError, ImportError) as e:
        return f"cute-dsl backend (MLA decode kernel) cannot implement this configuration: {e}"
    return None


@functools.cache
def _mla_decode_tuning_config(
    buckets: tuple[int, ...],
    num_pages: int,
    profile_seq_len: int,
    has_sparse_mla_top_k_lens: bool = False,
    sparse_top_k_width: int = 0,
    enable_dcp: bool = False,
    cp_world: int = 1,
    cp_rank: int = 0,
) -> TuningConfig:
    """One TuningConfig and stable initializer set per key.

    Memoized so equivalent dispatcher calls reuse a single config object (and
    its initializer closures) instead of rebuilding them every call. The
    initializer closures no longer participate in ``AutoTuner``'s nearest-profile
    cache key (it keys only on the dynamic-tensor specs and constraints), so this
    memoization is a host-overhead optimization rather than a leak guard.

    The DynamicTensorSpec sweeps batch dim across ``query``, ``block_tables``,
    ``seq_lens``, ``out``, plus one optional fifth tensor — the sparse top-k
    lengths tensor (native ``qk_rope_head_dim=0`` MLA) or, for DCP,
    ``causal_seqlens_kv_global``. ``block_tables`` is initialized via
    ``random_(0, num_pages)`` which wraps mod kv_cache size — safe for autotune
    profiling because MLA decode reads kv_cache and never writes it, so aliased
    page reads give correct timing measurements. ``seq_lens`` is filled with
    ``profile_seq_len``; the synthetic DCP global bound preserves that exact
    rank-local length.
    """

    def init_block_tables(shapes, dtype, device):
        tensor = torch.empty(shapes, dtype=dtype, device=device)
        tensor.random_(0, num_pages)
        return tensor

    def init_seq_lens(shapes, dtype, device):
        tensor = torch.empty(shapes, dtype=dtype, device=device)
        tensor.fill_(profile_seq_len)
        return tensor

    def init_sparse_top_k_lens(shapes, dtype, device):
        tensor = torch.empty(shapes, dtype=dtype, device=device)
        tensor.fill_(sparse_top_k_width)
        return tensor

    def init_causal_seqlens_kv_global(shapes, dtype, device):
        tensor = torch.empty(shapes, dtype=dtype, device=device)
        tensor.fill_(profile_seq_len * cp_world + cp_rank)
        return tensor

    # At most one optional fifth batch-swept tensor: native qk_rope_head_dim=0
    # sparse top-k lengths and DCP's global causal bound are mutually exclusive.
    if has_sparse_mla_top_k_lens:
        fifth_init = init_sparse_top_k_lens
    elif enable_dcp:
        fifth_init = init_causal_seqlens_kv_global
    else:
        fifth_init = None

    input_idx = (0, 1, 2, 3, 4) if fifth_init is not None else (0, 1, 2, 3)
    tensor_initializers = [
        (1, init_block_tables),
        (2, init_seq_lens),
    ]
    if fifth_init is not None:
        tensor_initializers.append((4, fifth_init))

    return TuningConfig(
        dynamic_tensor_specs=(
            DynamicTensorSpec(
                input_idx=input_idx,
                dim_idx=(0,) * len(input_idx),
                gen_tuning_buckets=buckets,
                map_to_tuning_buckets=make_bucket_mapper(buckets, round_map=False),
            ),
        ),
        tensor_initializers=tuple(tensor_initializers),
        use_cuda_graph=True,
        use_cold_l2_cache=True,
    )


def _build_mla_decode_tuning_config(
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    workspace_buffer: torch.Tensor,
    runner_names: Sequence[str],
    q_len: int,
    num_heads: int,
    kv_lora_rank: int,
    max_seq_len: int,
    device: torch.device,
    has_sparse_mla_top_k_lens: bool = False,
    cute_dsl_impl: str = "auto",
    sinks: Optional[
        Union[List[torch.Tensor], Tuple[torch.Tensor, ...], torch.Tensor]
    ] = None,
    enable_dcp: bool = False,
    cp_world: int = 1,
    cp_rank: int = 0,
) -> TuningConfig:
    """Reduce call args to the memoization key of ``_mla_decode_tuning_config``.

    Key stability: num_pages and profile_seq_len are fixed per KV-cache
    allocation. A caller that varies max_seq_len or block-table width per
    call gets one cached config per distinct value (bounded, unlike the
    per-call configs this replaces).
    """
    # kv_cache may be 3D [num_pages, page_size, D] or 4D
    # [num_pages, 1, page_size, D] after `_check_mla_functional_shape` —
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
        max_seq_len,
        device,
        cute_dsl_impl,
        sinks,
        enable_dcp,
    )

    return _mla_decode_tuning_config(
        buckets,
        num_pages,
        profile_seq_len,
        has_sparse_mla_top_k_lens,
        block_tables.shape[-1],
        enable_dcp,
        cp_world,
        cp_rank,
    )


def _validate_mla_ragged_query(
    query: torch.Tensor,
    seq_lens: torch.Tensor,
    cum_seq_lens_q: torch.Tensor,
    max_q_len: Optional[int],
) -> Tuple[int, int]:
    """Validate compact-Q metadata, reading it on the host only to infer its maximum."""
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
            raise ValueError("cum_seq_lens_q[-1] must match the flattened query length")
        q_lens = cum_seq_lens_q_host[1:] - cum_seq_lens_q_host[:-1]
        if torch.any(q_lens < 0).item():
            raise ValueError("cum_seq_lens_q must be monotonically non-decreasing")
        max_q_len = q_lens.max().item()
        if max_q_len <= 0:
            raise ValueError("cum_seq_lens_q must describe at least one query token")
    elif max_q_len <= 0:
        raise ValueError("max_q_len must be greater than 0")

    return batch_size, max_q_len


def _select_mla_ragged_backend(
    backend: str,
    trtllm_gen_not_supported_reason: Optional[str],
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    bmm1_scale: Union[float, torch.Tensor],
    bmm2_scale: Union[float, torch.Tensor],
    sinks: Optional[List[torch.Tensor]],
    sparse_mla_top_k: int,
    skip_softmax_threshold_scale_factor: Optional[float],
    uses_shared_paged_kv_idx: bool,
    qk_rope_head_dim: int,
    kv_lora_rank: int,
    is_var_seq: bool,
    cute_dsl_impl: str,
    cum_seq_lens_q: torch.Tensor,
    max_q_len: int,
    enable_dcp: bool,
    cp_world: int,
    lse_requested: bool,
) -> str:
    """Resolve direct ragged execution, probing CuTe only when it can be selected."""
    trt_var_q_reason = trtllm_gen_not_supported_reason

    # Explicit TRT and auto calls that TRT supports do not need to import
    # or validate CuTeDSL. Evaluate the fallback only when it can run.
    cute_dsl_reason: Optional[str] = None
    if backend == "cute-dsl" or (backend == "auto" and trt_var_q_reason is not None):
        cute_dsl_reason = _cute_dsl_incompatibility_reason(
            query,
            torch.bfloat16,
            bmm1_scale,
            bmm2_scale,
            sinks,
            sparse_mla_top_k,
            skip_softmax_threshold_scale_factor,
            uses_shared_paged_kv_idx,
            qk_rope_head_dim,
            kv_lora_rank,
            kv_cache.shape[-2],
            is_var_seq,
            cute_dsl_impl=cute_dsl_impl,
            cum_seq_lens_q=cum_seq_lens_q,
            max_q_len=max_q_len,
            enable_dcp=enable_dcp,
            cp_world=cp_world,
        )

    selected_var_q_backend: str
    if backend == "cute-dsl":
        if cute_dsl_reason is not None:
            raise ValueError(cute_dsl_reason)
        selected_var_q_backend = "cute-dsl"
    elif backend == "trtllm-gen":
        if trt_var_q_reason is not None:
            if lse_requested:
                raise NotImplementedError(trt_var_q_reason)
            raise ValueError(trt_var_q_reason)
        selected_var_q_backend = "trtllm-gen"
    else:
        # CuTeDSL fills TRT capability gaps and is the ragged-Q LSE path.
        if trt_var_q_reason is None:
            selected_var_q_backend = "trtllm-gen"
        elif cute_dsl_reason is None:
            selected_var_q_backend = "cute-dsl"
        else:
            raise ValueError(
                "auto: no backend supports this variable-Q configuration "
                f"(trtllm-gen: {trt_var_q_reason}; "
                f"cute-dsl: {cute_dsl_reason})"
            )

    return selected_var_q_backend


def _prepare_mla_uniform_lse(
    query: torch.Tensor, lse: Optional[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return a flat backend view and preserve the caller's LSE object for return."""
    user_lse = lse
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
    return lse, user_lse


# Shared public-request validation, cross-backend selection/autotuning and
# output/LSE return adaptation live here. Backend modules own their capability
# checks and extracted execution mechanics; remaining backend preparation can
# migrate separately without changing this dispatch boundary.
def _mla_with_kv_cache_impl(
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
    sparse_mla_top_k_lens: Optional[torch.Tensor] = None,
    enable_dcp: bool = False,
    cp_world: int = 1,
    cp_rank: int = 0,
    causal_seqlens_kv_global: Optional[torch.Tensor] = None,
    use_fp16_softmax: Optional[bool] = None,
    return_lse_base: Optional[Literal["basee", "base2"]] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Execute MLA decode or prefill through the selected backend:
    TRTLLM-GEN, CuteDSL, XQA, Cake, or SM120/SM121 sparse kernels.

    With ``backend="auto"``, SM100/SM103 devices use TRTLLM-GEN for sparse MLA
    when ``sparse_mla_top_k > 0``. SM120/SM121 devices use the packed sparse
    backend for ``sparse_mla_top_k > 0`` and XQA for dense decode.

    Parameters
    ----------
    query : torch.Tensor
        Query tensor with shape
        ``[batch_size, q_len_per_request, num_heads, head_dim_qk]`` where
        ``head_dim_qk = kv_lora_rank + qk_rope_head_dim``. When
        ``cum_seq_lens_q`` is provided, TRTLLM-GEN and monolithic CuTeDSL
        instead accept compact ``[total_q, num_heads, head_dim_qk]`` input.
        For the SM120/SM121 sparse backend, this must be BF16 with
        ``head_dim_qk == 576`` for v32/GLM_NSA or ``512`` for GLM-5.3 NoPE.
    kv_cache : torch.Tensor
        For TRTLLM-GEN, CuteDSL, and XQA, the paged KV cache is
        ``[num_pages, page_size, kv_lora_rank + qk_rope_head_dim]`` or
        ``[num_pages, 1, page_size, kv_lora_rank + qk_rope_head_dim]`` and uses
        the query-compatible dense dtype. For the SM120/SM121 v32/GLM sparse
        backend, this is a packed uint8 cache with 656 bytes per token, shaped
        ``[num_pages, page_size, 656]`` or ``[num_pages, 1, page_size, 656]``;
        for GLM-5.3 NoPE the payload is 528 bytes per token and padded rows
        (any last dim >= 528, including a legacy 656 pool) are accepted.
    workspace_buffer : torch.Tensor
        Pre-allocated workspace buffer. Must be zero-initialized on first use
        by kernels that use semaphore state.
    qk_nope_head_dim : int
        Non-RoPE query dimension before absorption. SM100/SM103 TRTLLM-GEN
        sparse MLA supports ``64`` with ``kv_lora_rank=512`` and
        ``qk_rope_head_dim=64``. The SM120/SM121 sparse v32/GLM backend
        validates it together with ``query.shape[-1]``. GLM-5.3 uses the
        native ``qk_rope_head_dim=0`` / ``query.shape[-1]=512`` geometry.
    kv_lora_rank : int
        Latent KV rank. TRTLLM-GEN and SM120/SM121 sparse v32/GLM use ``512``.
    qk_rope_head_dim : int
        RoPE head dimension. Sparse MLA paths use ``64``; the native
        no-RoPE TRTLLM-GEN path (``kv_lora_rank=512``) uses ``0``.
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
        Per-request physical KV sequence lengths for dense and TRTLLM-GEN
        paths. With DCP these are rank-local lengths and continue to control
        paging, memory bounds, and split-KV. For
        SM120/SM121 sparse v32/GLM, pass ``[batch_size, q_len_per_request]`` or
        flattened ``[batch_size * q_len_per_request]`` active top-k lengths; if
        ``None``, every column in ``block_tables`` is active.
    max_seq_len : int
        Maximum physical KV sequence length used for dense/TRTLLM-GEN
        scheduling. With DCP this is the maximum rank-local length.
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
        Blackwell, CuteDSL, XQA, and SM120/SM121 sparse v32/GLM require a float.
    bmm2_scale : Union[float, torch.Tensor]
        Fused scale for MLA BMM2. TRTLLM-GEN accepts a FP32 tensor or float.
        Blackwell, CuteDSL, and XQA require a float. SM120/SM121 sparse v32/GLM
        requires ``1.0``.
    sinks : Optional[List[torch.Tensor]]
        Additional value per head in the denominator of the softmax.
        Supported by ``cake``, ``trtllm-gen``, ``cute-dsl``, and ``sparse``.
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
        ``"cake"``, ``"trtllm-gen"``, ``"cute-dsl"``, and
        ``"sparse"``.
        ``"cake"`` explicitly selects the source-level Blackwell semantic
        dispatcher. It covers its qualified BF16/FP8 dense, sparse, compact-Q,
        sink, LSE, and caller-owned-output envelope. An untuned ``"auto"`` call
        never selects it; under ``flashinfer.autotune(True)`` its dense
        Blackwell route can win a uniform-Q ``"auto"`` request (see Autotune
        below). ``"auto"``
        chooses ``"trtllm-gen"`` for SM100/SM103 sparse MLA and chooses
        ``"sparse"`` for SM120/SM121 when ``sparse_mla_top_k > 0``; otherwise
        SM120/SM121 dense decode uses ``"xqa"``.
        For compact variable Q on SM100/SM103, ``"auto"`` keeps TRTLLM-GEN
        when it supports the call and uses monolithic CuTeDSL for TRT gaps or
        LSE output.
        The ``cute-dsl`` backend has monolithic and modular implementations
        with a common fixed-Q shape/dtype envelope. Compact variable Q is
        monolithic-only, while features such as ``sinks`` are modular-only;
        selection is controlled by the ``cute_dsl_impl`` kwarg below.
    is_var_seq : bool
        Whether the sequence length is variable.
        If True, the sequence length is variable.
        Otherwise,the sequence length is fixed for all the requests in the batch.
    uses_shared_paged_kv_idx : bool = True
        Whether K and V page indices are shared as a unified index.
        True (default) uses vLLM/FlashInfer layout with a 2D page table.
        False uses TRT-LLM layout with a 3D page table ``[batch_size, 2, max_num_pages_per_seq]``.
        False is only supported for trtllm-gen backend.
    use_fp16_softmax : Optional[bool]
        Select the trtllm-gen ``Fp16Softmax`` cubin variant. MLA decode is the
        primary consumer of this flag — `Fp16Softmax` generation cubins are
        only shipped for MLA head dims (``head_dim_qk/v ∈ {576/512, 320/256}``).
        When ``None`` (default) or ``False`` the standard FP32-accumulator
        softmax cubin is used. Only supported by ``backend="trtllm-gen"``;
        passing ``True`` to other backends raises ``ValueError``.
    lse : Optional[torch.Tensor] = None
        Optional pre-allocated buffer for Log-Sum-Exp values. Supported by
        ``cake``, ``trtllm-gen``, ``cute-dsl``, and ``sparse`` backends. Must
        have dtype ``torch.float32``. Accepted shapes:

        * ``[batch_size * q_len_per_request, num_qo_heads]`` (TRTLLM-GEN
          native; accepted by sparse), or
        * ``[batch_size, q_len_per_request, num_qo_heads]`` (cute-dsl native;
          also accepted by cute-dsl), or
        * ``[total_q, num_qo_heads]`` for compact variable Q with monolithic
          CuTeDSL.

        If ``return_lse`` is True and this is None, a buffer will be
        allocated by the backend.
    return_lse : bool = False
        Whether to return LSE values. Supported by ``cake``, ``trtllm-gen``,
        ``cute-dsl``, and ``sparse`` backends. When True, the function
        returns ``(out, lse)``. With compact variable Q, LSE is currently
        supported by monolithic CuTeDSL and by the ``cake`` Kimi-K3 FP8 route.
    return_lse_base : Optional[Literal["basee", "base2"]] = None
        Logarithm base of LSE values, including values written to a supplied
        ``lse`` buffer. Supported by ``trtllm-gen``, ``cute-dsl``, ``sparse`` and
        the ``cake`` Kimi-K3 FP8 route. ``"basee"`` selects natural-log units and
        ``"base2"`` selects base-2 units. ``None`` preserves each backend's
        historical default: base-2 for ``trtllm-gen`` and ``sparse``, and
        natural-log for monolithic ``cute-dsl`` and the ``cake`` Kimi-K3 route.
        Select an explicit base for stable units under
        ``backend="auto"``. This option does not enable LSE output; use
        ``return_lse=True`` to request it. Other values raise
        :class:`ValueError`.
    cute_dsl_impl : str = "auto"
        Which cute-dsl implementation to use. Honored when
        ``backend="cute-dsl"`` and when ``backend="auto"`` considers the
        cute-dsl candidate; ignored for non-cute-dsl backends.

        * ``"auto"`` (default) — picks monolithic by default, automatically
          promoted to modular when the call uses a feature monolithic
          doesn't support (currently ``sinks``), and keeps compact variable Q
          on monolithic.
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
        and is monotonically non-decreasing. Supported by Blackwell, TRTLLM-GEN,
        and the monolithic CuTeDSL implementation. When provided, ``query``
        must have shape ``[total_q, num_heads, head_dim_qk]``.
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
    sparse_mla_top_k_lens : Optional[torch.Tensor] = None
        Flattened active sparse top-k lengths, one INT32 value per query token.
        Required by the native ``kv_lora_rank=512, qk_rope_head_dim=0``
        TRTLLM-GEN kernel. Sparse indices must be packed before any ``-1``
        padding. Zero-length rows are unsupported; padded query rows should
        contain one valid dummy index and use length 1.
    enable_dcp : bool = False
        Statically enable cyclic decode context parallelism in the monolithic
        CuTeDSL MLA kernel (``backend="cute-dsl"`` or ``"auto"``) or in the
        ``backend="cake"`` Kimi-K3 FP8 route (power-of-two ``cp_world``). DCP
        returns a rank-local output/LSE state, so ``return_lse=True`` is
        required and the caller must merge rank states. Both fixed-Q input and
        compact variable-Q input described by ``cum_seq_lens_q`` are supported.
    cp_world : int = 1
        Compile-time context-parallel world size. Rank ``r`` stores global KV
        positions whose token index modulo ``cp_world`` equals ``r``.
    cp_rank : int = 0
        Runtime-uniform context-parallel rank.
    causal_seqlens_kv_global : Optional[torch.Tensor] = None
        Contiguous CUDA int32 tensor ``[batch_size]`` containing the global
        exclusive causal bound for the newest query token. Required with DCP.

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
    On SM100/SM103/SM107 dense MLA, calling under ``flashinfer.autotune(True)`` with
    ``backend="auto"`` profiles both ``trtllm-gen`` and ``cute-dsl`` when DCP
    is disabled. DCP profiles only its required monolithic CuTeDSL runner.
    Both modes use a bucketed batch sweep up to each runner's kernel/workspace
    cap and cache the winning tactic per shape signature. Subsequent calls under
    ``autotune(False)`` use the cached choice; any batch outside the tuned range
    falls back to the runner's default tactic with a one-time warning.
    On SM100a/SM103a, Cake's generated Blackwell route (the one
    ``backend="cake"`` takes for its dimension tuples) can win ``backend="auto"``
    in a second selection stage. The first stage is the selection above,
    unchanged: the same op (``"trtllm_batch_decode_mla"``), runners, buckets,
    cache entries and replay records as without Cake, tuned and looked up first,
    so every bucket holds its TRTLLM-GEN / CuTe DSL winner and every lookup
    selects what it selects without Cake (a repeated lookup-only call of a
    listed batch reuses that result, see below). The second stage, the op
    ``"trtllm_batch_decode_mla_cake_margin"``, compares that winner with Cake per
    tuning bucket, under the same ``autotune()`` settings, and (outside
    ``autotune_v2`` contexts) keeps Cake unless the first-stage winner measures
    faster by more than ``tie_tolerance`` or by a tie of about 2.6 us (1.25
    steps of the timer's resolution) at the first stage's time of the winner,
    whichever is wider (a relative tolerance, so it scales with the second
    stage's measurement of the winner); it profiles Cake only against a
    first-stage winner, and a Cake result is valid only while the first stage
    keeps the winner Cake beat or tied. The second stage runs only for the units the device's offer table
    (``_CAKE_MLA_AUTO_OFFERS``: device, dtype, page size, ``q_len``,
    ``max_seq_len`` bucket and batch range) lists, where Cake measured faster
    than the other candidates; elsewhere, and on devices the table does not
    list, Cake is never offered. Requests with sinks are not offered. Outside
    tuning the second stage runs only while the tuner holds a Cake result, and
    tuning under ``flashinfer.autotuner.set_autotune_process_group`` skips it.
    Cake is eligible only for dense requests its aligned programs serve with
    the same semantics: BF16 query and cache with page size 32 or 64, or FP8
    E4M3 query and cache with page size 64 and no sink; 128 heads of a generated
    dimension tuple, host float scales, shared K/V page indices and contiguous
    tensors; never with sparse MLA, ``sparse_mla_top_k_lens``, skip-softmax,
    ``use_fp16_softmax``, DCP, a ``multi_ctas_kv_counter_buffer`` or LSE output.
    Its generic tail program and the Kimi-K3 FP8 Cake route never join. Cake
    serves at most 512 query rows (``batch * q_len``; FP8 128) and is offered for
    a tuning bucket only when every batch the bucket serves is listed and
    launches the programs of the profiled batch, so never for the largest
    bucket. An untuned call never selects Cake. Every Cake miss runs the first
    stage's selection: a cached Cake choice that cannot launch for the actual
    batch, whose first-stage winner changed, or whose program is not yet loaded
    during CUDA-graph capture, a failure while tuning Cake, and every request
    or setting that leaves Cake out, including
    ``FLASHINFER_DISABLE_CAKE_AUTO_CANDIDATES=1``. ``autotune_v2`` replay
    diagnostics record the first stage only; under strict replay a missing
    second-stage entry runs the first stage's selection. Explicit
    ``backend="cake"`` is unaffected. Unlike explicit ``backend="cake"``, the
    candidate never reads ``seq_lens`` on the host and keeps no per-input plan
    or cached scratch, so in-place input updates and CUDA-graph replay with
    updated inputs stay correct; it never takes Cake's exact-length programs.
    Its FP8 page-64 route allocates its split-K scratch (about 256 KiB per query
    row) on every call. Padded rows (``seq_lens`` 0, or a query row with no
    causal KV token) output zeros. Tuning measures GPU time under CUDA-graph
    replay; an eager caller also pays the host dispatch. Outside tuning, a
    batch no listed unit covers skips the second stage, and for a listed batch
    the decision of both stages (also the first stage's alone, where Cake is
    left out because the tuner holds no Cake result (nor, with
    ``FLASHINFER_AUTOTUNER_LOAD_FROM_FILE=1``, the bundled configs) or it is
    switched off) is
    reused for repeated calls with the same arguments and tensor sizes while
    the tuner, the offer table, every tuner's winners and loaded configs, the
    ``autotune()`` overrides, ``skip_ops``,
    ``FLASHINFER_AUTOTUNER_LOAD_FROM_FILE`` and
    ``FLASHINFER_DISABLE_CAKE_AUTO_CANDIDATES`` are unchanged (not while
    tuning, with an ``autotune_v2`` store or context, under replay diagnostics
    or with a measurement policy): such a call neither checks whether Cake
    joins, nor builds the tuning config, nor looks up, and builds only the
    selected runner; a reused Cake decision checks the contiguity of the
    tensors and that its programs are loaded, and launches the programs it
    resolved for those sizes. A ``max_seq_len`` counts as the same argument
    while its power-of-two bucket, its 128-token KV tile count and whether it
    reaches the device's split-KV range are the same (the decision reads
    nothing else of it). A call with new arguments
    under the same tuner state (such as a ``max_seq_len`` of another bucket)
    selects, but reuses the last check of whether Cake joins.
    Uniform explicit CuTeDSL requests use the same tuner with one candidate.
    Explicit TRTLLM-GEN requests enter the tuner only during tuning; inference
    directly executes its sole tactic. Compact variable-Q requests select an eligible backend directly
    and do not enter the autotuner. The internal ``autotune`` execution route
    does not introduce a public functional ``backend="autotune"`` option.

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
    # Shared argument validation. Packed sparse inputs keep their own contract.
    if return_lse_base is not None and (
        not isinstance(return_lse_base, str)
        or return_lse_base not in ("basee", "base2")
    ):
        raise ValueError(
            "return_lse_base must be 'basee', 'base2', or None; "
            f"got {return_lse_base!r}"
        )
    if isinstance(bmm1_scale, torch.Tensor):
        if bmm1_scale.dtype != torch.float32:
            raise TypeError("bmm1_scale tensor must have dtype torch.float32")
    if isinstance(bmm2_scale, torch.Tensor):
        if bmm2_scale.dtype != torch.float32:
            raise TypeError("bmm2_scale tensor must have dtype torch.float32")
    if max_q_len is not None and cum_seq_lens_q is None:
        raise ValueError("max_q_len is only supported when cum_seq_lens_q is provided")

    is_nope_mla = (
        kv_lora_rank == nope_mla_dimensions.kv_lora_rank
        and qk_rope_head_dim == nope_mla_dimensions.qk_rope_head_dim
    )
    if sparse_mla_top_k_lens is not None:
        if not is_nope_mla:
            raise ValueError(
                "sparse_mla_top_k_lens is currently only supported by the "
                "native qk_rope_head_dim=0 TRTLLM-GEN MLA path"
            )
        expected_num_query_tokens = (
            query.size(0) * query.size(1) if query.ndim == 4 else query.size(0)
        )
        check_shape_dtype_device(
            sparse_mla_top_k_lens,
            (expected_num_query_tokens,),
            torch.int32,
            query.device,
            "sparse_mla_top_k_lens",
        )
        sparse_mla_top_k_lens = sparse_mla_top_k_lens.contiguous()

    if kv_cache.dtype == torch.uint8 and sparse_mla_top_k <= 0:
        raise NotImplementedError(
            "Dense MLA decode does not support packed uint8 KV caches yet: no "
            "backend has an NVFP4 MLA decode kernel. The NVFP4 MLA cache write "
            "path is available via nvfp4_quantize_append_paged_mla_kv_cache."
        )

    resolved_backend = _validate_mla_dcp_args(
        query=query,
        backend=backend,
        sinks=sinks,
        cum_seq_lens_q=cum_seq_lens_q,
        max_q_len=max_q_len,
        return_lse=return_lse,
        enable_dcp=enable_dcp,
        cp_world=cp_world,
        cp_rank=cp_rank,
        causal_seqlens_kv_global=causal_seqlens_kv_global,
    )

    if resolved_backend not in (
        "auto",
        "trtllm-gen",
        "cute-dsl",
        "xqa",
        "sparse",
        "cake",
    ):
        raise ValueError(f"Backend {resolved_backend} not supported")

    check_trtllm_gen_sm107_only_feature(
        use_fp16_softmax, "use_fp16_softmax", query.device
    )

    # Resolve architecture policy before applying backend-specific contracts.
    if resolved_backend == "auto":
        cc = get_compute_capability(query.device)
        if cc[0] == 12 and sparse_mla_top_k > 0:
            resolved_backend = "sparse"
        elif cc[0] != 10:
            resolved_backend = "xqa"

    # The native no-rope trtllm-gen/cute-dsl kernels require the per-token
    # active top-k length; the SM120 sparse backend bounds each row by its
    # -1 entries instead and does not consume sparse_mla_top_k_lens.
    if is_nope_mla and resolved_backend not in ("sparse", "cake"):
        if sparse_mla_top_k <= 0:
            raise ValueError(
                "Native qk_rope_head_dim=0 TRTLLM-GEN MLA requires sparse_mla_top_k > 0"
            )
        if sparse_mla_top_k_lens is None:
            raise ValueError(
                "Native qk_rope_head_dim=0 TRTLLM-GEN MLA requires sparse_mla_top_k_lens"
            )
    if sparse_mla_top_k_lens is not None and resolved_backend == "sparse":
        raise ValueError(
            "sparse_mla_top_k_lens is not supported by the SM120 sparse MLA "
            "backend; pass per-token active top-k lengths via seq_lens instead"
        )

    # Resolve eligibility and fallback without constructing or launching runners.
    # TRT/CuTe share input normalization. Ragged eligibility precedes shape checks,
    # whereas uniform eligibility follows them, preserving existing diagnostics.
    if resolved_backend in ("auto", "trtllm-gen", "cute-dsl"):
        if seq_lens is None:
            raise ValueError(
                "seq_lens is required for trtllm-gen and cute-dsl MLA backends"
            )

        # Shared setup for the trtllm-gen / cute-dsl autotune dispatch.
        enable_pdl = (
            device_support_pdl(query.device) if enable_pdl is None else enable_pdl
        )
        sm_count = get_device_sm_count(query.device)

        block_size = kv_cache.size(-2)
        num_heads_q = query.size(-2)
        trtllm_gen_not_supported_reason = _trtllm_gen_mla_incompatibility_reason(
            num_heads_q=num_heads_q,
            block_size=block_size,
            has_var_q=cum_seq_lens_q is not None,
            lse_requested=return_lse or lse is not None,
        )

        if skip_softmax_threshold_scale_factor is not None and sparse_mla_top_k != 0:
            raise ValueError("skip_softmax is not supported for sparse MLA")

        has_var_q = cum_seq_lens_q is not None
        if has_var_q:
            batch_size, max_q_len = _validate_mla_ragged_query(
                query, seq_lens, cum_seq_lens_q, max_q_len
            )
            resolved_backend = _select_mla_ragged_backend(
                resolved_backend,
                trtllm_gen_not_supported_reason,
                query,
                kv_cache,
                bmm1_scale,
                bmm2_scale,
                sinks,
                sparse_mla_top_k,
                skip_softmax_threshold_scale_factor,
                uses_shared_paged_kv_idx,
                qk_rope_head_dim,
                kv_lora_rank,
                is_var_seq,
                cute_dsl_impl,
                cum_seq_lens_q,
                max_q_len,
                enable_dcp,
                cp_world,
                return_lse or lse is not None,
            )
            kv_cache = _check_mla_functional_shape(
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
                require_aligned_block_table=resolved_backend == "cute-dsl",
            )

            out = _prepare_trtllm_gen_mla_output(
                out, query.shape[:-1] + (kv_lora_rank,), query.device
            )

            if return_lse:
                expected_lse_shape = (query.size(0), query.size(1))
                if lse is None:
                    lse = torch.empty(
                        expected_lse_shape,
                        dtype=torch.float32,
                        device=query.device,
                    )
                else:
                    check_shape_dtype_device(
                        lse,
                        expected_lse_shape,
                        torch.float32,
                        query.device,
                        "lse",
                    )

        else:
            # Explicit TRT accepts native page-table width; auto and CuTe require alignment.
            kv_cache = _check_mla_functional_shape(
                query,
                kv_cache,
                kv_lora_rank,
                qk_rope_head_dim,
                sparse_mla_top_k,
                block_tables,
                block_size,
                uses_shared_paged_kv_idx,
                require_aligned_block_table=resolved_backend != "trtllm-gen",
            )

            # Pre-allocate `out` so non-swept dims have a template for autotune
            # profiling (the autotuner inherits non-swept dims from caller tensors).
            out = _prepare_trtllm_gen_mla_output(
                out, query.shape[:-1] + (kv_lora_rank,), query.device
            )

            user_lse = lse
            if return_lse:
                lse, user_lse = _prepare_mla_uniform_lse(query, lse)

            page_size = kv_cache.shape[-2]
            cute_dsl_reason = None
            if resolved_backend != "trtllm-gen":
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
                    use_fp16_softmax=use_fp16_softmax,
                    cute_dsl_impl=cute_dsl_impl,
                    enable_dcp=enable_dcp,
                    cp_world=cp_world,
                )
            # Batch ranges of the units for which the second selection stage
            # offers Cake (``_mla_cake_stage_two``); empty: no second stage.
            cake_batch_ranges: Tuple[Tuple[int, int], ...] = ()
            # For a lookup-only call of a listed batch: the memo key, the
            # tuner state and offer table read before both stages, and the
            # decision ``_CAKE_MLA_AUTO_MEMO`` holds for them.
            cake_memo_key: Optional[tuple] = None
            cake_memo_state: Optional[tuple] = None
            cake_offers: Any = None
            cake_decision: Optional[tuple] = None
            tuner = AutoTuner.get()
            if resolved_backend == "cute-dsl":
                if cute_dsl_reason is not None:
                    raise ValueError(cute_dsl_reason)
                runner_names = ["cute-dsl"]
            elif resolved_backend == "trtllm-gen":
                if trtllm_gen_not_supported_reason is not None:
                    raise ValueError(trtllm_gen_not_supported_reason)
                runner_names = ["trtllm-gen"]
            else:  # resolved_backend == "auto"
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
                # The runners above are the first selection stage, as without
                # Cake. Cake joins a second stage only for the units the device
                # lists, only while the tuner may select it, and never while an
                # autotune process group tunes (rank-local builds and buckets
                # could differ per rank). Outside tuning only a batch the units
                # list joins it; tuning joins every batch, so one tuning call
                # times every offered bucket, as the first stage does.
                # A lookup-only call of a listed batch reuses the decision of
                # both stages memoized for its arguments and tuner state
                # (``_CAKE_MLA_AUTO_MEMO``) instead of the Cake candidate gate,
                # the eligibility check and both lookups: the gate is a
                # function of the state.
                # Read once: a decision is memoized only while the table it
                # was made under is the offer table.
                cake_offers = _CAKE_MLA_AUTO_OFFERS
                # The memoized ranges and the batch come first: an unlisted
                # unit, or a batch outside its ranges outside tuning, pays for
                # nothing else.
                device_index = query.get_device()
                offered = _cake_mla_auto_batch_ranges(
                    device_index,
                    query.dtype,
                    page_size,
                    query.size(1),
                    sinks is not None,
                    max_seq_len,
                )
                if (
                    offered
                    and (
                        tuner.is_tuning_mode or _in_batch_ranges(query.size(0), offered)
                    )
                    # Before the memo key and the gate: such calls never join
                    # Cake and pay for neither.
                    and _cake_mla_auto_scalar_reason(
                        bmm1_scale=bmm1_scale,
                        bmm2_scale=bmm2_scale,
                        sparse_mla_top_k=sparse_mla_top_k,
                        sparse_mla_top_k_lens=sparse_mla_top_k_lens,
                        skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
                        lse_requested=return_lse or lse is not None,
                        use_fp16_softmax=use_fp16_softmax,
                        enable_dcp=enable_dcp,
                        multi_ctas_kv_counter_buffer=multi_ctas_kv_counter_buffer,
                        uses_shared_paged_kv_idx=uses_shared_paged_kv_idx,
                    )
                    is None
                ):
                    if not tuner.is_tuning_mode:
                        cake_memo_state = _mla_cake_memo_state(tuner)
                    if cake_memo_state is not None:
                        try:
                            # Every argument the first stage's tuning config,
                            # the two stages' lookups and the Cake checks
                            # read, except the contiguity of the tensors and
                            # the sinks (none: ``offered`` is empty with
                            # sinks), and except the arguments the scalar
                            # check above fixed; see the audit at
                            # ``_CAKE_MLA_AUTO_MEMO``. Of ``max_seq_len`` only
                            # what the decision reads, so a caller varying it
                            # selects once per bucket, KV tile count and side
                            # of the split-KV range. The
                            # sign of ``bmm2_scale`` keeps -0.0 and 0.0 apart
                            # (the launch plan holds it).
                            cake_memo_key = (
                                device_index,
                                query.shape,
                                query.dtype,
                                kv_cache.shape,
                                kv_cache.dtype,
                                block_tables.shape,
                                block_tables.dtype,
                                seq_lens.shape,
                                seq_lens.dtype,
                                out.shape,
                                out.dtype,
                                workspace_buffer.shape,
                                workspace_buffer.dtype,
                                _mla_auto_memo_seq_len(max_seq_len, device_index),
                                bmm1_scale,
                                bmm2_scale,
                                math.copysign(1.0, bmm2_scale),
                                qk_nope_head_dim,
                                kv_lora_rank,
                                qk_rope_head_dim,
                                sparse_mla_top_k,
                                use_fp16_softmax,
                                enable_pdl,
                                is_var_seq,
                                tuple(runner_names),
                                cute_dsl_impl,
                                return_lse_base,
                                cp_world,
                                cp_rank,
                            )
                            cake_decision = _mla_cake_auto_memo_get(
                                tuner,
                                cake_memo_key,
                                cake_memo_state,
                                query,
                                kv_cache,
                                block_tables,
                                seq_lens,
                                out,
                            )
                        except TypeError:  # an unhashable argument
                            cake_memo_key = None
                    # On a miss the gate decides as without the memo (with a
                    # state, memoized for the last state). Closed, the
                    # decision is the first stage's selection, memoized under
                    # the key like a rejection the key decides.
                    if cake_decision is None and (
                        cake_auto_candidate_open(_CAKE_MLA_RUNNER)
                        if cake_memo_state is None
                        else _mla_cake_candidate_open(tuner, cake_memo_state)
                    ):
                        reason = _cake_mla_auto_candidate_reason(
                            query=query,
                            kv_cache=kv_cache,
                            block_tables=block_tables,
                            seq_lens=seq_lens,
                            out=out,
                            qk_nope_head_dim=qk_nope_head_dim,
                            kv_lora_rank=kv_lora_rank,
                            qk_rope_head_dim=qk_rope_head_dim,
                            sparse_mla_top_k=sparse_mla_top_k,
                            bmm1_scale=bmm1_scale,
                            bmm2_scale=bmm2_scale,
                            sinks=sinks,
                            skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
                            uses_shared_paged_kv_idx=uses_shared_paged_kv_idx,
                            lse_requested=return_lse or lse is not None,
                            use_fp16_softmax=use_fp16_softmax,
                            multi_ctas_kv_counter_buffer=multi_ctas_kv_counter_buffer,
                            sparse_mla_top_k_lens=sparse_mla_top_k_lens,
                            enable_dcp=enable_dcp,
                        )
                        if reason is None:
                            cake_batch_ranges = offered
                        elif reason in (
                            _CAKE_MLA_SINK_REASON,
                            _CAKE_MLA_CONTIGUITY_REASON,
                        ):
                            # Not decided by the key: nothing is memoized.
                            cake_memo_key = None
                        # Else the key decides the rejection: the first
                        # stage's selection is memoized.

            resolved_backend = "autotune"

    # Execute the resolved route. Selection policy and candidate execution remain separate.
    if resolved_backend == "autotune":
        if (
            multi_ctas_kv_counter_buffer is not None
            and "trtllm-gen" not in runner_names
        ):
            raise ValueError(
                "multi_ctas_kv_counter_buffer is only supported when a trtllm-gen runner is selected"
            )
        if multi_ctas_kv_counter_buffer is not None:
            multi_ctas_kv_counter_buffer = (
                _resolve_trtllm_gen_multi_ctas_kv_counter_buffer(
                    multi_ctas_kv_counter_buffer,
                    query.size(0),
                    query.size(2),
                    sm_count,
                    query.device,
                )
            )

        # A memoized decision builds only the runner it selects.
        built_names: Sequence[str] = runner_names
        if cake_decision is not None:
            built_names = (
                (runner_names[cake_decision[0]],) if cake_decision[2] is None else ()
            )
        runners: List[TunableRunner] = []
        if "trtllm-gen" in built_names:
            # The TRT adapter lowers tensor scales once before profiling/launch.
            runners.append(
                _BatchMLAPagedAttentionTrtllmGenBackend.from_functional(
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
                    return_lse_base=return_lse_base,
                    use_fp16_softmax=use_fp16_softmax,
                )
            )
        if "cute-dsl" in built_names:
            # Normalize sinks: public contract accepts Optional[List[Tensor]] for
            # legacy reasons, but cute-dsl's modular variant expects a single
            # per-head tensor or None. The list-of-1 case has been guarded by
            # `_cute_dsl_incompatibility_reason` so we can unpack here safely.
            cute_dsl_sinks: Optional[torch.Tensor] = None
            if sinks is not None:
                cute_dsl_sinks = sinks[0] if isinstance(sinks, (list, tuple)) else sinks
            from ..cute_dsl.attention.mla_dispatch import _resolve_impl

            resolved_impl = _resolve_impl(
                requested=cute_dsl_impl,
                kwargs={"sinks": cute_dsl_sinks, "enable_dcp": enable_dcp},
            )
            # Each resolved CuTe implementation owns its functional adapter.
            cute_runner_factory = (
                _BatchMLAPagedAttentionCuteDslMonolithicBackend.from_functional
                if resolved_impl == "monolithic"
                else _BatchMLAPagedAttentionCuteDslModularBackend.from_functional
            )
            runners.append(
                cute_runner_factory(
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
                    return_lse_base=return_lse_base,
                    sinks=cute_dsl_sinks,
                    cute_dsl_impl=cute_dsl_impl,
                    enable_dcp=enable_dcp,
                    cp_world=cp_world,
                    cp_rank=cp_rank,
                )
            )

        inputs = [query, block_tables, seq_lens, out]
        if sparse_mla_top_k_lens is not None:
            inputs.append(sparse_mla_top_k_lens)
        elif enable_dcp:
            # The global causal bound varies with batch and must be synthesized
            # alongside the other batch-shaped tensors during autotuning.
            inputs.append(causal_seqlens_kv_global)
        if cake_decision is not None:
            # Both stages' decision, memoized for this call's arguments.
            _, tactic, cake_choice = cake_decision
            if cake_choice is None:
                runner = runners[0]
            else:
                cake = _CakeTrtllmMlaBlackwellRunner(
                    kv_cache=kv_cache,
                    qk_nope_head_dim=qk_nope_head_dim,
                    kv_lora_rank=kv_lora_rank,
                    qk_rope_head_dim=qk_rope_head_dim,
                    max_seq_len=max_seq_len,
                    bmm1_scale=cast(float, bmm1_scale),
                    bmm2_scale=cast(float, bmm2_scale),
                    sinks=sinks,
                    enable_pdl=enable_pdl,
                )
                cake._plan = cake_choice[2]
                runner, tactic = cake, cake_choice[0]
        elif backend == "trtllm-gen" and not tuner.is_tuning_mode:
            # Explicit TRT has one implementation and only tactic -1. Keep
            # profiling inside autotune(), but inference needs no selection.
            runner, tactic = runners[0], -1
        else:
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
                has_sparse_mla_top_k_lens=sparse_mla_top_k_lens is not None,
                cute_dsl_impl=cute_dsl_impl,
                sinks=sinks,
                enable_dcp=enable_dcp,
                cp_world=cp_world,
                cp_rank=cp_rank,
            )
            runner, tactic = tuner.choose_one(
                _MLA_DECODE_OP,
                runners,
                tuning_config,
                inputs,
            )
            if cake_batch_ranges:
                selected = (runner, tactic)
                memoizable: Optional[List[Optional[tuple]]] = (
                    None if cake_memo_key is None else []
                )
                runner, tactic = _mla_cake_stage_two(
                    tuner,
                    runners,
                    tuning_config,
                    inputs,
                    selected,
                    # Built only when the second stage needs it. A partial,
                    # not a closure: the locals of this function stay plain
                    # locals (no cells) on every call of every backend.
                    functools.partial(
                        _CakeTrtllmMlaBlackwellRunner,
                        kv_cache=kv_cache,
                        qk_nope_head_dim=qk_nope_head_dim,
                        kv_lora_rank=kv_lora_rank,
                        qk_rope_head_dim=qk_rope_head_dim,
                        max_seq_len=max_seq_len,
                        bmm1_scale=cast(float, bmm1_scale),
                        bmm2_scale=cast(float, bmm2_scale),
                        sinks=sinks,
                        enable_pdl=enable_pdl,
                    ),
                    cake_batch_ranges,
                    None,
                    memoizable,
                    cake_memo_state,
                )
                if memoizable:
                    _mla_cake_auto_memo_put(
                        tuner,
                        cast(tuple, cake_memo_key),
                        cast(tuple, cake_memo_state),
                        cake_offers,
                        runners,
                        selected,
                        memoizable[0],
                    )
            elif cake_memo_key is not None:
                # The eligibility check rejected Cake for a reason the key
                # decides: the decision is the first stage's selection.
                _mla_cake_auto_memo_put(
                    tuner,
                    cake_memo_key,
                    cast(tuple, cake_memo_state),
                    cake_offers,
                    runners,
                    (runner, tactic),
                    None,
                )
        if isinstance(runner, _BatchMLAPagedAttentionTrtllmGenBackend):
            runner(
                inputs=inputs,
                tactic=tactic,
                multi_ctas_kv_counter_buffer=multi_ctas_kv_counter_buffer,
            )
        else:
            runner(inputs=inputs, tactic=tactic)
        # Return the original 2D/3D caller object, or the allocated flat default.
        return (out, user_lse) if return_lse else out
    elif resolved_backend == "trtllm-gen":
        # Validate caller storage before acquiring the executable, preserving
        # public error ordering. The runner owns allocation when none is supplied.
        if multi_ctas_kv_counter_buffer is not None:
            multi_ctas_kv_counter_buffer = (
                _resolve_trtllm_gen_multi_ctas_kv_counter_buffer(
                    multi_ctas_kv_counter_buffer,
                    batch_size,
                    query.size(1),
                    sm_count,
                    query.device,
                )
            )
        runner = _BatchMLAPagedAttentionTrtllmGenBackend.from_functional(
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
            return_lse=False,
            lse=None,
            return_lse_base=return_lse_base,
            use_fp16_softmax=use_fp16_softmax,
        )
        inputs = [query, block_tables, seq_lens, out]
        if sparse_mla_top_k_lens is not None:
            inputs.append(sparse_mla_top_k_lens)
        return runner(
            inputs=inputs,
            tactic=-1,
            multi_ctas_kv_counter_buffer=multi_ctas_kv_counter_buffer,
            cum_seq_lens_q=cum_seq_lens_q,
            max_q_len=max_q_len,
        )

    elif resolved_backend == "cute-dsl":
        if multi_ctas_kv_counter_buffer is not None:
            raise ValueError(
                "multi_ctas_kv_counter_buffer is only supported by the "
                "trtllm-gen backend"
            )
        # Compact variable-Q has already resolved to monolithic CuTe. Execute
        # the same backend directly, without functional batch-bucket tuning.
        runner = _BatchMLAPagedAttentionCuteDslMonolithicBackend.from_functional(
            kv_cache=kv_cache,
            workspace_buffer=workspace_buffer,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            max_seq_len=max_seq_len,
            softmax_scale=bmm1_scale,
            output_scale=bmm2_scale,
            out_dtype=out.dtype,
            is_var_seq=is_var_seq,
            enable_pdl=enable_pdl,
            uses_shared_paged_kv_idx=uses_shared_paged_kv_idx,
            lse=lse,
            return_lse=return_lse,
            return_lse_base=return_lse_base,
            sinks=None,
            cute_dsl_impl=cute_dsl_impl,
            enable_dcp=enable_dcp,
            cp_world=cp_world,
            cp_rank=cp_rank,
        )
        inputs = [query, block_tables, seq_lens, out]
        if enable_dcp:
            inputs.append(causal_seqlens_kv_global)
        return runner(
            inputs=inputs,
            tactic=-1,
            cum_seq_lens_q=cum_seq_lens_q,
            max_q_len=max_q_len,
        )

    elif resolved_backend == "xqa":
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
                "XQA MLA requires SM120a (CUDA >= 12.8) or SM121a (CUDA >= 12.9)"
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
        if use_fp16_softmax:
            raise ValueError(
                "use_fp16_softmax is only supported by backend='trtllm-gen'"
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
    elif resolved_backend == "cake":
        if not enable_dcp and _cake_trtllm_mla_blackwell_supports(
            query, qk_nope_head_dim, kv_lora_rank, qk_rope_head_dim, sparse_mla_top_k
        ):
            # Two Cake MLA families answer to backend="cake": the TRT-LLM-style Blackwell decode
            # programs for their generated dimension tuples, and the Kimi-K3 FP8 paged-cache route
            # (below) for everything else in its contract.
            from .cake_trtllm_mla_blackwell import trtllm_mla_blackwell_decode

            return trtllm_mla_blackwell_decode(
                query=query,
                kv_cache=kv_cache,
                block_tables=block_tables,
                seq_lens=seq_lens,
                max_seq_len=max_seq_len,
                qk_nope_head_dim=qk_nope_head_dim,
                kv_lora_rank=kv_lora_rank,
                qk_rope_head_dim=qk_rope_head_dim,
                sparse_mla_top_k=sparse_mla_top_k,
                out=out,
                bmm1_scale=bmm1_scale,
                bmm2_scale=bmm2_scale,
                sinks=sinks,
                skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
                enable_pdl=enable_pdl,
                uses_shared_paged_kv_idx=uses_shared_paged_kv_idx,
                lse=lse,
                return_lse=return_lse,
                cum_seq_lens_q=cum_seq_lens_q,
                max_q_len=max_q_len,
                multi_ctas_kv_counter_buffer=multi_ctas_kv_counter_buffer,
                sparse_mla_top_k_lens=sparse_mla_top_k_lens,
                enable_dcp=enable_dcp,
                backend="cake",
            )

        # Cake-generated Kimi-K3 MLA over the FP8 paged latent cache (SM100 / SM103): dense
        # decode, packed variable-Q / MTP and incremental prefill; BF16 output, caller-owned
        # ``out``, current stream, CUDA-Graph replayable; FP32 LSE (natural log unless
        # ``return_lse_base="base2"``) and cyclic DCP (``causal_seqlens_kv_global``, ``cp_world``,
        # ``cp_rank``).  Unsupported here: sparse top-k, sinks, skip-softmax, FP16 softmax, PDL,
        # NVFP4 / uint8 caches.
        unsupported = []
        if sparse_mla_top_k > 0 or sparse_mla_top_k_lens is not None:
            unsupported.append("sparse_mla_top_k")
        if sinks is not None:
            unsupported.append("sinks")
        if skip_softmax_threshold_scale_factor is not None:
            unsupported.append("skip_softmax_threshold_scale_factor")
        if use_fp16_softmax:
            unsupported.append("use_fp16_softmax")
        if enable_pdl:
            unsupported.append("enable_pdl")
        if multi_ctas_kv_counter_buffer is not None:
            unsupported.append("multi_ctas_kv_counter_buffer")
        if unsupported:
            raise ValueError(
                "backend='cake' does not support " + ", ".join(unsupported)
            )
        if seq_lens is None:
            raise ValueError("backend='cake' requires seq_lens")
        if query.dtype != torch.float8_e4m3fn or kv_cache.dtype != torch.float8_e4m3fn:
            raise ValueError(
                "backend='cake' requires float8_e4m3fn query and kv_cache, got "
                f"{query.dtype} and {kv_cache.dtype}"
            )
        if kv_lora_rank != 512 or qk_rope_head_dim != 64:
            raise ValueError(
                "backend='cake' supports kv_lora_rank=512 and qk_rope_head_dim=64 only"
            )
        if isinstance(bmm1_scale, torch.Tensor) or isinstance(bmm2_scale, torch.Tensor):
            raise ValueError("backend='cake' takes host float bmm1_scale / bmm2_scale")
        if out is None:
            out = torch.empty(
                (*query.shape[:-1], kv_lora_rank),
                dtype=torch.bfloat16,
                device=query.device,
            )
        if return_lse and lse is None:
            lse = torch.empty(
                query.shape[:-1], dtype=torch.float32, device=query.device
            )
        from .cake_kimi_k3_mla import run_cake_kimi_k3_mla_fp8_paged_attention

        run_cake_kimi_k3_mla_fp8_paged_attention(
            query,
            kv_cache,
            block_tables,
            seq_lens,
            out,
            workspace_buffer,
            bmm1_scale=float(bmm1_scale),
            bmm2_scale=float(bmm2_scale),
            cum_seq_lens_q=cum_seq_lens_q,
            max_q_len=max_q_len,
            max_seq_len=int(max_seq_len),
            lse=lse,
            lse_base="2" if return_lse_base == "base2" else "e",
            causal_seqs=causal_seqlens_kv_global
            if enable_dcp and cp_world > 1
            else None,
            cp_world=cp_world if enable_dcp else 1,
            cp_rank=cp_rank if enable_dcp else 0,
        )
        return (out, lse) if return_lse else out

    elif resolved_backend == "sparse":
        # Packed SM120/SM121 sparse MLA only, despite the helper's legacy
        # "_trtllm_" name. TRTLLM-GEN dense and native sparse MLA use the
        # TRT runner paths above.
        if multi_ctas_kv_counter_buffer is not None:
            raise ValueError(
                "multi_ctas_kv_counter_buffer is only supported by the trtllm-gen backend"
            )
        lse_scale = 1.0
        if return_lse_base == "basee":
            lse_scale = 1.0 / log2e
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
            lse_scale=lse_scale,
            kv_scale_format=kv_scale_format,
        )

    raise AssertionError(f"Unresolved MLA route: {resolved_backend}")


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
    sparse_mla_top_k_lens: Optional[torch.Tensor] = None,
    enable_dcp: bool = False,
    cp_world: int = 1,
    cp_rank: int = 0,
    causal_seqlens_kv_global: Optional[torch.Tensor] = None,
    use_fp16_softmax: Optional[bool] = None,
    return_lse_base: Optional[Literal["basee", "base2"]] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """See :func:`_mla_with_kv_cache_impl` for parameter documentation."""
    return _mla_with_kv_cache_impl(
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
        sparse_mla_top_k_lens=sparse_mla_top_k_lens,
        enable_dcp=enable_dcp,
        cp_world=cp_world,
        cp_rank=cp_rank,
        causal_seqlens_kv_global=causal_seqlens_kv_global,
        use_fp16_softmax=use_fp16_softmax,
        return_lse_base=return_lse_base,
    )


trtllm_batch_decode_with_kv_cache_mla.__doc__ = _mla_with_kv_cache_impl.__doc__


@flashinfer_experimental_api
def prepare_nvfp4_batch_decode_with_kv_cache_mla(
    query: torch.Tensor,
    query_scale: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_scale: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    workspace_buffer: torch.Tensor,
    *,
    sm_scale: float,
    sinks: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    return_lse: bool = False,
    seq_lens_cpu: Optional[torch.Tensor] = None,
    backend: str = "cake",
):
    """Prepare NVFP4 DeepSeek-V4 paged MQA decode attention.

    The experimental Cake backend serves ``[batch * q_len, num_heads, 512]``
    queries (``q_len`` query tokens per request, derived from the query
    rows; six for DeepSeek-V4) against a shared paged K/V cache of 64-token
    pages, both stored as packed E2M1 bytes (256 per row) with UE4M3
    block-16 scales (32 per row), with a causal mask inside each request's
    query block, optional per-head attention sinks, BF16 output and
    natural-log FP32 LSE. It requires compute capability 10.0 or 10.3 and at
    least 128 packed query rows per batch (``batch * q_len * num_heads``).

    Preparation validates inputs, builds the host work plan from the
    sequence lengths (one device-to-host copy unless ``seq_lens_cpu`` is
    given), carves split partials out of ``workspace_buffer`` and returns an
    ``NVFP4MLADecodeRunner``. Calling the runner launches the decode kernel
    and, when planned, the split-KV combine kernel without CUDA allocation and
    returns ``out`` (or ``(out, lse)`` with ``return_lse=True``). Prepare a
    new runner after changing sequence lengths, bindings or input values.
    CUDA Graph ownership remains with the caller. See
    ``flashinfer/experimental/nvfp4_mla_decode/README.md``.
    """
    if backend != "cake":
        raise ValueError("NVFP4 MLA decode currently supports backend='cake'")
    from ..experimental.nvfp4_mla_decode.cake_backend import (
        prepare_nvfp4_batch_decode_with_kv_cache_mla as prepare,
    )

    return prepare(
        query,
        query_scale,
        kv_cache,
        kv_scale,
        block_tables,
        seq_lens,
        workspace_buffer,
        sm_scale=sm_scale,
        sinks=sinks,
        out=out,
        lse=lse,
        return_lse=return_lse,
        seq_lens_cpu=seq_lens_cpu,
        backend="cake",
    )


@flashinfer_experimental_api(feature="NVFP4 sparse MLA decode (SM100/SM103)")
def nvfp4_sparse_mla_decode(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    bmm1_scale: float,
    bmm2_scale: float = 1.0,
    out: Optional[torch.Tensor] = None,
    num_ctas_per_token: Optional[int] = None,
    backend: str = "cuda",
) -> torch.Tensor:
    r"""Sparse MLA decode over an NVFP4 (``nvfp4_ds_mla``) KV cache on SM100 and SM103.

    Each query token attends to the KV rows its sparse indexer selected (DeepSeek-V3.2 and GLM-5 DSA)::

        out[t, h] = bmm2_scale * sum_k softmax_k(bmm1_scale * query[t, h] . K[i_tk]) * V[i_tk]

    ``K`` is a row's 576-dim latent (512 NoPE + 64 RoPE) and ``V`` its first 512 dims. Rows are dequantized exactly
    in the kernel; nothing is staged in global memory.

    Parameters
    ----------
    query : torch.Tensor
        ``[num_tokens, 16, 576]`` ``torch.float8_e4m3fn``: 16 heads per rank (for example DeepSeek-V3.2 at TP8 or
        GLM-5 at TP4).
    kv_cache : torch.Tensor
        ``torch.uint8`` rows of 352 bytes in vLLM's ``nvfp4_ds_mla`` layout, any leading shape (for example
        ``[num_blocks, block_size, 352]``). Bytes 0-255 hold the 512 NoPE values as e2m1 (element ``2i`` in the
        low nibble), bytes 256-319 the 64 RoPE values as e4m3, bytes 320-351 one e4m3 scale per 16 NoPE values,
        block ``b``'s at byte ``320 + 8 * (b % 4) + b // 4``.
    indices : torch.Tensor
        ``[num_tokens, topk]`` ``torch.int32`` row ids into ``kv_cache`` viewed as ``[-1, 352]``
        (``block_id * block_size + offset``); ``-1`` marks an empty slot. ``topk`` is a multiple of 32 (2048 in
        DeepSeek-V3.2 and GLM-5). Values are not range-checked.
    bmm1_scale : float
        Scale of ``query . K`` before the softmax: the softmax scale times the query's dequantization scale.
    bmm2_scale : float
        Scale of the output.
    out : Optional[torch.Tensor]
        ``[num_tokens, 16, 512]`` ``torch.bfloat16`` output; allocated when omitted.
    num_ctas_per_token : Optional[int]
        Thread-block cluster size per query token: 3 to 8 with ``backend="cuda"``, 1 to 6 or 8 with
        ``backend="cake"`` (7 is rejected). By default, among the sizes whose ``num_tokens`` clusters all fit
        on the device in one wave, ``backend="cuda"`` takes the largest and ``backend="cake"`` the one with
        the least work per CTA.
    backend : str
        ``"cuda"`` (default), the hand-written kernel, or ``"cake"``, the generated program of the same
        operator (``flashinfer/experimental/nvfp4_sparse_mla_decode/cake_backend.py``).

    Returns
    -------
    torch.Tensor
        ``out``. A token whose indices are all ``-1`` gets zeros.

    Notes
    -----
    Requires compute capability 10.0 (B200, GB200) or 10.3 (B300, GB300). The first call on a device compiles
    the kernel and queries its cluster occupancy; later calls can be captured in CUDA graphs. Layout, limits and measurements:
    ``flashinfer/experimental/nvfp4_sparse_mla_decode/README.md``.
    """
    if backend == "cake":
        from ..experimental.nvfp4_sparse_mla_decode.cake_backend import (
            nvfp4_sparse_mla_decode as run_cake,
        )

        return run_cake(
            query,
            kv_cache,
            indices,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            out=out,
            num_ctas_per_token=num_ctas_per_token,
            backend="cake",
        )
    if backend != "cuda":
        raise ValueError(
            "nvfp4_sparse_mla_decode supports backend='cuda' (default) or backend='cake'"
        )
    from ..experimental.nvfp4_sparse_mla_decode.backend import run

    return run(
        query,
        kv_cache,
        indices,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        out=out,
        num_ctas_per_token=num_ctas_per_token,
    )


@flashinfer_experimental_api(feature="Cake MLA variable-query DCP decode")
def cake_mla_varq_dcp_decode(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    workspace_buffer: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    softmax_scale: float,
    *,
    cum_seq_lens_q: torch.Tensor,
    max_q_len: int,
    enable_dcp: bool = False,
    cp_world: int = 1,
    cp_rank: int = 0,
    causal_seqlens_kv_global: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    return_lse: bool = True,
    backend: str = "cake",
):
    """Compact variable-query MLA decode of one decode-context-parallel rank.

    The experimental Cake backend serves DeepSeek MLA decode (576 = 512 latent
    + 64 rope per key, 512-wide value) over a rank-local paged BF16 or FP8
    (e4m3) KV cache with 32-, 64- or 128-token pages, compact variable-length
    queries ``[total_q, num_heads, 576]`` (``cum_seq_lens_q`` / ``max_q_len``,
    ``num_heads <= 128``) and the static cyclic DCP visibility rule of
    ``cute_dsl_mla_decode(..., is_var_seq=True, enable_dcp=True)``: rank
    ``cp_rank`` of ``cp_world`` holds the global positions ``cp_world * k +
    cp_rank`` and ``causal_seqlens_kv_global`` bounds each request.  It writes
    BF16 ``out [total_q, num_heads, 512]`` and natural-log FP32 ``lse
    [total_q, num_heads]`` (``O = 0`` / ``LSE = -inf`` for rows without a
    visible key) and requires compute capability 10.0 or 10.3.  The caller
    owns ``workspace_buffer`` (uint8; size from
    ``flashinfer.experimental.cake_mla_varq_dcp_decode.cake_backend
    .cake_mla_varq_dcp_decode_workspace_size``).  ``max_seq_len`` is the
    caller-known largest rank-local length; nothing reads device tensor
    contents on the host.  Returns ``(out, lse)`` with ``return_lse=True``.
    ``prepare_cake_mla_varq_dcp_decode`` returns a launch-only runner for
    repeated calls with fixed bindings.  See
    ``tests/experimental/test_cake_mla_varq_dcp_decode.py`` for the validated
    shape set and ``benchmarks/bench_cake_mla_varq_dcp_decode.py`` for the
    comparison against ``cute_dsl_mla_decode``.
    """
    if backend != "cake":
        raise ValueError("Cake MLA var-Q DCP decode currently supports backend='cake'")
    from ..experimental.cake_mla_varq_dcp_decode.cake_backend import (
        cake_mla_varq_dcp_decode as run,
    )

    return run(
        query,
        kv_cache,
        workspace_buffer,
        block_tables,
        seq_lens,
        max_seq_len,
        softmax_scale,
        cum_seq_lens_q=cum_seq_lens_q,
        max_q_len=max_q_len,
        enable_dcp=enable_dcp,
        cp_world=cp_world,
        cp_rank=cp_rank,
        causal_seqlens_kv_global=causal_seqlens_kv_global,
        out=out,
        lse=lse,
        return_lse=return_lse,
        backend="cake",
    )


@flashinfer_experimental_api(feature="Prepared Cake MLA variable-query DCP decode")
def prepare_cake_mla_varq_dcp_decode(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    cum_seq_lens_q: torch.Tensor,
    max_q_len: int,
    *,
    max_seq_len: int,
    softmax_scale: float,
    workspace_buffer: torch.Tensor,
    causal_seqlens_kv_global: Optional[torch.Tensor] = None,
    cp_world: int = 1,
    cp_rank: int = 0,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    backend: str = "cake",
):
    """Plan and bind one Cake compact var-Q (+DCP) MLA decode problem.

    Validation, the host plan, the workspace carve and every allocation happen
    here; the returned ``CakeMLAVarQDcpDecodeRunner`` launches the persistent
    decode kernel (and, when the plan splits items, the split-KV merge kernel
    behind it) with no CUDA allocation and no host synchronization and returns
    ``(out, lse)``.  Prepare a new runner when shapes, ``max_seq_len`` or the
    tensor bindings change; never share one workspace between two live
    runners.  CUDA Graph ownership remains with the caller.  See
    ``cake_mla_varq_dcp_decode`` for the semantics.
    """
    if backend != "cake":
        raise ValueError("Cake MLA var-Q DCP decode currently supports backend='cake'")
    from ..experimental.cake_mla_varq_dcp_decode.cake_backend import (
        prepare_cake_mla_varq_dcp_decode as prepare,
    )

    return prepare(
        query,
        kv_cache,
        page_table,
        seq_lens,
        cum_seq_lens_q,
        max_q_len,
        max_seq_len=max_seq_len,
        softmax_scale=softmax_scale,
        workspace_buffer=workspace_buffer,
        causal_seqlens_kv_global=causal_seqlens_kv_global,
        cp_world=cp_world,
        cp_rank=cp_rank,
        out=out,
        lse=lse,
        backend="cake",
    )


_CAKE_NVFP4_MLA_RUBIN_CAPABILITY = (10, 7)


def _cake_nvfp4_mla_is_rubin(device: torch.device) -> bool:
    """Whether the Cake NVFP4 MLA decode entry points route to the Rubin (SM107)
    package for ``device``; every other device goes to the SM100 / SM103
    package, which raises its own error for an unsupported capability."""
    return (
        device.type == "cuda"
        and tuple(get_compute_capability(device)) == _CAKE_NVFP4_MLA_RUBIN_CAPABILITY
    )


def _cake_nvfp4_mla_check_rubin_only(cp_interleave: int) -> None:
    if cp_interleave != 1:
        raise ValueError(
            "cp_interleave other than 1 is served by the Rubin (SM107) Cake NVFP4 MLA "
            "decode only; the SM100 / SM103 package shards keys one at a time"
        )


@flashinfer_experimental_api(feature="Cake dense NVFP4 MLA decode (SM100/SM103/SM107)")
def cake_mla_nvfp4_paged_decode(
    q_nope: torch.Tensor,
    q_sf: torch.Tensor,
    q_rope: torch.Tensor,
    q_scale: torch.Tensor,
    ckv_cache: torch.Tensor,
    ckv_sf_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    workspace_buffer: torch.Tensor,
    *,
    sm_scale: float,
    ckv_scale: float,
    o_scale: float = 1.0,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    return_lse: bool = False,
    cum_seq_lens_q: Optional[torch.Tensor] = None,
    max_q_len: Optional[int] = None,
    max_seq_len: Optional[int] = None,
    cp_world: int = 1,
    cp_rank: int = 0,
    kv_len_global: Optional[torch.Tensor] = None,
    backend: str = "cake",
    seq_lens_host: Optional[Sequence[int]] = None,
    kv_len_global_host: Optional[Sequence[int]] = None,
    cum_seq_lens_q_host: Optional[Sequence[int]] = None,
    cp_interleave: int = 1,
    units: Optional[int] = None,
    pairs: Optional[int] = None,
):
    """Dense absorbed-MLA decode over an NVFP4 paged latent cache.

    The experimental Cake backend serves Kimi-K3 / DeepSeek-V3 MLA decode
    (512 latent + 64 rope channels per key, 512-wide value) over the NVFP4
    paged cache of flashinfer-ai/flashinfer#4676: ``ckv_cache`` uint8
    ``[num_pages, page_size, 256]`` packed E2M1, ``ckv_sf_cache``
    float8_e4m3fn ``[num_pages, page_size, 32]`` block-16 scales,
    ``kpe_cache`` float8_e4m3fn ``[num_pages, page_size, 64]`` rope channels
    (page sizes 32 / 64 / 128 / ... on SM100 / SM103, 64 on SM107; strided
    views of one allocation allowed), decoded as ``e2m1 * sf * ckv_scale |
    fp8 * kpe_scale``.  The query is NVFP4 too -- ``q_nope`` uint8
    ``[.., 256]``, ``q_sf`` float8_e4m3fn ``[.., 32]``, ``q_rope``
    float8_e4m3fn ``[.., 64]``, ``q_scale`` float32 ``[..]`` from
    :func:`quantize_mla_nvfp4_query` -- as ``[batch, q_len, num_heads, .]``
    or packed ``[total_q, num_heads, .]`` with ``cum_seq_lens_q`` /
    ``max_q_len`` (variable-length / MTP queries, bottom-right causal against
    the request's last keys).  It writes BF16 ``out`` (``q.shape[:-1] +
    (512,)``) and, with ``return_lse=True``, the natural-log FP32 ``lse``;
    decode context parallelism follows the static cyclic rule (``cp_world``
    / ``cp_rank`` with the per-request global lengths ``kv_len_global``;
    SM107 also takes ``cp_interleave``, the block of consecutive keys per
    rank).  Requires compute capability 10.0, 10.3 or 10.7: the call routes
    on the device's capability to ``flashinfer.experimental.
    cake_mla_nvfp4_paged_decode`` (SM100 / SM103) or ``flashinfer.
    experimental.cake_rubin_mla_nvfp4_decode`` (SM107).  The caller owns
    ``workspace_buffer`` (uint8; size from the package's ``cake_backend
    .workspace_bytes`` / ``max_workspace_bytes``).  On SM100 / SM103 nothing
    reads device tensor contents on the host; the SM107 host plan reads the
    KV and query lengths, from ``seq_lens_host`` / ``cum_seq_lens_q_host`` /
    ``kv_len_global_host`` when given, else by copying the device tensors
    (a synchronization), and takes the device's cluster counts ``units`` /
    ``pairs`` (queried when omitted).  ``prepare_cake_mla_nvfp4_paged_decode``
    returns a launch-only runner for repeated calls with fixed bindings.  See
    the two packages' ``README.md``.
    """
    if backend != "cake":
        raise ValueError("Cake NVFP4 MLA decode currently supports backend='cake'")
    if _cake_nvfp4_mla_is_rubin(q_nope.device):
        from ..experimental.cake_rubin_mla_nvfp4_decode.cake_backend import (
            cake_mla_nvfp4_paged_decode as run_rubin,
        )

        return run_rubin(
            q_nope,
            q_sf,
            q_rope,
            q_scale,
            ckv_cache,
            ckv_sf_cache,
            kpe_cache,
            block_tables,
            seq_lens,
            workspace_buffer,
            sm_scale=sm_scale,
            ckv_scale=ckv_scale,
            o_scale=o_scale,
            out=out,
            lse=lse,
            return_lse=return_lse,
            cum_seq_lens_q=cum_seq_lens_q,
            max_q_len=max_q_len,
            max_seq_len=max_seq_len,
            cp_world=cp_world,
            cp_rank=cp_rank,
            kv_len_global=kv_len_global,
            backend="cake",
            seq_lens_host=seq_lens_host,
            kv_len_global_host=kv_len_global_host,
            cum_seq_lens_q_host=cum_seq_lens_q_host,
            cp_interleave=cp_interleave,
            units=units,
            pairs=pairs,
        )
    _cake_nvfp4_mla_check_rubin_only(cp_interleave)
    from ..experimental.cake_mla_nvfp4_paged_decode.cake_backend import (
        cake_mla_nvfp4_paged_decode as run,
    )

    return run(
        q_nope,
        q_sf,
        q_rope,
        q_scale,
        ckv_cache,
        ckv_sf_cache,
        kpe_cache,
        block_tables,
        seq_lens,
        workspace_buffer,
        sm_scale=sm_scale,
        ckv_scale=ckv_scale,
        o_scale=o_scale,
        out=out,
        lse=lse,
        return_lse=return_lse,
        cum_seq_lens_q=cum_seq_lens_q,
        max_q_len=max_q_len,
        max_seq_len=max_seq_len,
        cp_world=cp_world,
        cp_rank=cp_rank,
        kv_len_global=kv_len_global,
        backend="cake",
    )


@flashinfer_experimental_api(
    feature="Prepared Cake dense NVFP4 MLA decode (SM100/SM103/SM107)"
)
def prepare_cake_mla_nvfp4_paged_decode(
    q_nope: torch.Tensor,
    q_sf: torch.Tensor,
    q_rope: torch.Tensor,
    q_scale: torch.Tensor,
    ckv_cache: torch.Tensor,
    ckv_sf_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    out: torch.Tensor,
    workspace_buffer: torch.Tensor,
    *,
    sm_scale: float,
    ckv_scale: float,
    o_scale: float = 1.0,
    lse: Optional[torch.Tensor] = None,
    cum_seq_lens_q: Optional[torch.Tensor] = None,
    max_q_len: Optional[int] = None,
    max_seq_len: Optional[int] = None,
    cp_world: int = 1,
    cp_rank: int = 0,
    kv_len_global: Optional[torch.Tensor] = None,
    backend: str = "cake",
    seq_lens_host: Optional[Sequence[int]] = None,
    kv_len_global_host: Optional[Sequence[int]] = None,
    cum_seq_lens_q_host: Optional[Sequence[int]] = None,
    cp_interleave: int = 1,
    units: Optional[int] = None,
    pairs: Optional[int] = None,
):
    """Plan and bind one Cake dense NVFP4 MLA decode problem.

    Validation, the host plan (SM100 / SM103: row tile, KV split count,
    grids, reducer; SM107: the work-item table and per-row key limits from
    the host lengths), the workspace carve and every allocation happen here;
    the returned runner launches the attention kernel (and, when the plan
    splits the KV, the merge kernel behind it) with no CUDA allocation and no
    host synchronization, writing the caller-owned ``out`` / ``lse``.  Prepare
    a new runner when shapes, lengths, ``max_seq_len`` or the tensor bindings
    change.  CUDA Graph ownership remains with the caller; a captured
    ``launch()`` replays through the runner's workspace, so the runner and
    ``workspace_buffer`` stay alive while the graph is replayed.  The call routes on
    the device's compute capability (10.7: the Rubin package, which takes the
    host length lists and ``cp_interleave`` / ``units`` / ``pairs``; 10.0 /
    10.3: the SM100 / SM103 package).  See :func:`cake_mla_nvfp4_paged_decode`
    for the semantics.
    """
    if backend != "cake":
        raise ValueError("Cake NVFP4 MLA decode currently supports backend='cake'")
    if _cake_nvfp4_mla_is_rubin(q_nope.device):
        from ..experimental.cake_rubin_mla_nvfp4_decode.cake_backend import (
            prepare_cake_mla_nvfp4_paged_decode as prepare_rubin,
        )

        return prepare_rubin(
            q_nope,
            q_sf,
            q_rope,
            q_scale,
            ckv_cache,
            ckv_sf_cache,
            kpe_cache,
            block_tables,
            seq_lens,
            out,
            workspace_buffer,
            sm_scale=sm_scale,
            ckv_scale=ckv_scale,
            o_scale=o_scale,
            lse=lse,
            cum_seq_lens_q=cum_seq_lens_q,
            max_q_len=max_q_len,
            max_seq_len=max_seq_len,
            cp_world=cp_world,
            cp_rank=cp_rank,
            kv_len_global=kv_len_global,
            backend="cake",
            seq_lens_host=seq_lens_host,
            kv_len_global_host=kv_len_global_host,
            cum_seq_lens_q_host=cum_seq_lens_q_host,
            cp_interleave=cp_interleave,
            units=units,
            pairs=pairs,
        )
    _cake_nvfp4_mla_check_rubin_only(cp_interleave)
    from ..experimental.cake_mla_nvfp4_paged_decode.cake_backend import (
        CakeMlaNvfp4PagedDecode,
    )

    return CakeMlaNvfp4PagedDecode(
        q_nope=q_nope,
        q_sf=q_sf,
        q_rope=q_rope,
        q_scale=q_scale,
        ckv_cache=ckv_cache,
        ckv_sf_cache=ckv_sf_cache,
        kpe_cache=kpe_cache,
        block_tables=block_tables,
        seq_lens=seq_lens,
        out=out,
        workspace_buffer=workspace_buffer,
        sm_scale=sm_scale,
        ckv_scale=ckv_scale,
        o_scale=o_scale,
        lse=lse,
        cum_seq_lens_q=cum_seq_lens_q,
        max_q_len=max_q_len,
        max_seq_len=max_seq_len,
        cp_world=cp_world,
        cp_rank=cp_rank,
        kv_len_global=kv_len_global,
    )


@flashinfer_experimental_api(
    feature="Cake NVFP4 MLA query quantization (SM100/SM103/SM107)"
)
def quantize_mla_nvfp4_query(
    query: torch.Tensor,
    ckv_scale: float,
    kpe_scale: float,
    *,
    out: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a BF16 MLA query ``[.., 576]`` into the NVFP4 operands of
    :func:`cake_mla_nvfp4_paged_decode`: ``(q_nope, q_sf, q_rope, q_scale)``.

    Per (token, head) row ``q_scale = max(amax(nope) / (6 * 448),
    amax(rope) * kpe_scale / (448 * ckv_scale))``, E4M3 block-16 scales
    ``amax_block / (6 q_scale)``, E2M1 codes ``x / (sf q_scale)`` (ties to
    even) and FP8 rope ``x * kpe_scale / (q_scale * ckv_scale)``, so the rope
    product lands in the latent logit unit.  ``out`` reuses caller buffers
    (``cake_backend.mla_nvfp4_query_buffers``); the launch then allocates
    nothing.  Requires compute capability 10.0, 10.3 or 10.7 (the call
    routes on the device's capability; the two packages' quantizers are bit
    for bit the same arithmetic).
    """
    if _cake_nvfp4_mla_is_rubin(query.device):
        from ..experimental.cake_rubin_mla_nvfp4_decode.cake_backend import (
            quantize_mla_nvfp4_query as run_rubin,
        )

        return run_rubin(query, ckv_scale, kpe_scale, out=out)
    from ..experimental.cake_mla_nvfp4_paged_decode.cake_backend import (
        quantize_mla_nvfp4_query as run,
    )

    return run(query, ckv_scale, kpe_scale, out=out)


@flashinfer_api(trace=trtllm_batch_decode_mla_trace_dispatch)
def trtllm_prefill_with_kv_cache_mla(
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
    sparse_mla_top_k_lens: Optional[torch.Tensor] = None,
    enable_dcp: bool = False,
    cp_world: int = 1,
    cp_rank: int = 0,
    causal_seqlens_kv_global: Optional[torch.Tensor] = None,
    use_fp16_softmax: Optional[bool] = None,
    return_lse_base: Optional[Literal["basee", "base2"]] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Run MLA with decode-compatible semantics for prefill callers.

    This function shares its implementation, argument and return contract,
    exceptions, and backend selection with
    :func:`trtllm_batch_decode_with_kv_cache_mla`. Use
    ``q_len_per_request == 1`` for decode, or use a compatible backend with
    ``q_len_per_request > 1`` for incremental prefill and multi-token
    prediction (MTP).

    XQA only supports ``q_len_per_request == 1``. Because ``backend="auto"``
    may select XQA, callers that require multi-token prefill must select a
    compatible backend explicitly.

    This entrypoint is also available as
    :func:`flashinfer.prefill.trtllm_prefill_with_kv_cache_mla`. See
    :func:`trtllm_batch_decode_with_kv_cache_mla` for complete parameter and
    return-value documentation.
    """
    return _mla_with_kv_cache_impl(
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
        sparse_mla_top_k_lens=sparse_mla_top_k_lens,
        enable_dcp=enable_dcp,
        cp_world=cp_world,
        cp_rank=cp_rank,
        causal_seqlens_kv_global=causal_seqlens_kv_global,
        use_fp16_softmax=use_fp16_softmax,
        return_lse_base=return_lse_base,
    )


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
    backend = _BatchMLAPagedAttentionXqaBackend.from_functional(
        query=query,
        kv_cache=kv_cache,
        workspace_buffer=workspace_buffer,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        block_tables=block_tables,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        sinks=sinks,
        enable_pdl=enable_pdl,
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

    tuner = AutoTuner.get()
    # XQA currently has one implementation and one tactic. Record measurements
    # when requested, but the inference choice is always the same native launch.
    if not tuner.is_tuning_mode:
        return backend._execute(
            query,
            backend._kv_cache,
            block_tables,
            seq_lens.unsqueeze(1),
            out,
            bmm1_scale,
            bmm2_scale,
        )
    inputs = [query, block_tables, seq_lens, out]
    if isinstance(bmm1_scale, torch.Tensor) or isinstance(bmm2_scale, torch.Tensor):
        inputs.extend((bmm1_scale, bmm2_scale))
    runner, tactic = tuner.choose_one(
        "xqa_batch_decode_mla", [backend], _XQA_MLA_TUNING_CONFIG, inputs
    )
    return runner(inputs, tactic=tactic)
