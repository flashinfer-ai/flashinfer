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
from typing import List, Literal, Optional, Tuple, Union, overload

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


def _check_trtllm_gen_mla_shape(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    sparse_mla_top_k: int,
    page_table: torch.Tensor,
    page_size: int,
    uses_shared_paged_kv_idx: bool = True,
) -> torch.Tensor:
    if query.ndim != 4:
        raise ValueError(f"Expected query.ndim == 4, got {query.ndim}")

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

    num_seqs, num_tokens, _, qk_head_dim = query.shape
    ckv_dim = kv_cache.shape[3]
    expected_qk_head_dim = kv_lora_rank + qk_rope_head_dim
    if qk_head_dim != expected_qk_head_dim or ckv_dim != expected_qk_head_dim:
        raise ValueError(
            f"Expected head dim {expected_qk_head_dim} for query and kv_cache, got {qk_head_dim} and {ckv_dim}"
        )

    if sparse_mla_top_k > 0:
        page_table_shape = page_table.shape
        if page_table_shape != (num_seqs, num_tokens, sparse_mla_top_k):
            raise ValueError(
                f"Expected page_table.shape == (num_seqs, num_tokens, sparse_mla_top_k), got {page_table_shape}"
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
        if block_num % (128 / block_size) != 0:
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

    swa_kv_cache = _normalize_dsv4_sparse_mla_kv_cache(
        swa_kv_cache, kv_layout, "swa_kv_cache"
    )
    if swa_kv_cache.dtype != query.dtype:
        raise ValueError(
            f"swa_kv_cache dtype must match query dtype, got {swa_kv_cache.dtype} "
            f"and {query.dtype}"
        )
    if swa_kv_cache.size(-1) != 512:
        raise ValueError(
            f"Expected swa_kv_cache head dim 512, got {swa_kv_cache.size(-1)}"
        )

    compressed_kv_cache = _normalize_dsv4_sparse_mla_kv_cache(
        compressed_kv_cache, kv_layout, "compressed_kv_cache"
    )
    if compressed_kv_cache.dtype != query.dtype:
        raise ValueError(
            "compressed_kv_cache dtype must match query dtype, got "
            f"{compressed_kv_cache.dtype} and {query.dtype}"
        )
    if compressed_kv_cache.size(-1) != 512:
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


def trtllm_batch_decode_sparse_mla_dsv4(
    query: torch.Tensor,
    swa_kv_cache: torch.Tensor,
    workspace_buffer: torch.Tensor,
    sparse_indices: torch.Tensor,
    compressed_kv_cache: torch.Tensor,
    sparse_topk_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    bmm1_scale: Union[float, torch.Tensor] = 1.0,
    bmm2_scale: Union[float, torch.Tensor] = 1.0,
    sinks: Optional[torch.Tensor] = None,
    kv_layout: Literal["HND", "NHD"] = "HND",
    cum_seq_lens_q: Optional[torch.Tensor] = None,
    max_q_len: Optional[int] = None,
    enable_pdl: bool | None = None,
) -> torch.Tensor:
    r"""Decode DeepSeek V4 sparse MLA with separate SWA and compressed KV pools.

    This API is for the TRTLLM-GEN DeepSeek V4 sparse MLA kernels where
    ``headDimQk == headDimV == 512``. It supports BF16 and per-tensor FP8 E4M3
    inputs with BF16 output. The SWA side is fixed to 128 indices per query.
    Callers must provide a primary/compressed KV pool, the concatenated sparse
    index matrix, and total sparse MLA top-k lengths for every query token.

    Parameters
    ----------
    query : torch.Tensor
        Dense query input ``[batch_size, q_len_per_request, num_heads, 512]``
        or varlen query input ``[sum_q, num_heads, 512]`` when
        ``cum_seq_lens_q`` is provided. BF16 or FP8 E4M3.
    swa_kv_cache : torch.Tensor
        SWA KV cache. HND layout is ``[num_pages, 1, page_size, 512]`` and NHD
        layout is ``[num_pages, page_size, 1, 512]``. A 3D HND shorthand
        ``[num_pages, page_size, 512]`` is also accepted.
    workspace_buffer : torch.Tensor
        TRTLLM-GEN workspace buffer. Must be zero-initialized for first use.
    sparse_indices : torch.Tensor
        Flattened concatenated sparse MLA physical token indices in query-token
        order with shape ``[sum_q, sparse_topk_capacity]``. The first 128
        columns are SWA indices into ``swa_kv_cache``; the remaining columns are
        compressed/top-k indices into ``compressed_kv_cache``. For SWA-only
        calls, callers may provide padded invalid compressed columns, while
        ``sparse_topk_lens`` controls the active length.
    compressed_kv_cache : torch.Tensor
        Primary/compressed KV cache in the same layout and dtype as
        ``swa_kv_cache``. For SWA-only calls, provide any valid primary pool.
    sparse_topk_lens : torch.Tensor
        Flattened total sparse MLA top-k lengths in query-token order, shape
        ``[sum_q]``. Values must already include the fixed 128 SWA entries,
        matching TRTLLM-GEN ``sparseMlaTopkLengths``, and must not exceed
        ``sparse_indices.shape[-1]``.
    seq_lens : torch.Tensor
        Original KV sequence lengths for the SWA side, shape ``[batch_size]``
        INT32. This is required even though ``sparse_topk_lens`` already
        includes the fixed SWA 128 entries. TRTLLM-GEN uses ``seq_lens`` and the
        per-request query length to compute the valid length of tile 0, the
        SWA-128 tile, as ``min(seq_lens[b] - q_lens[b] + q_idx + 1, 128)``.
        ``sparse_topk_lens`` controls the total sparse MLA length; it does not
        replace the SWA tile validity signal.
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
        Maximum query length in the varlen batch. Required when
        ``cum_seq_lens_q`` is provided so the wrapper does not need a
        device-to-host synchronization to infer it.
    enable_pdl : Optional[bool]
        Whether to enable Programmatic Dependent Launch.
    """
    if enable_pdl is None:
        enable_pdl = device_support_pdl(query.device)
    if isinstance(bmm1_scale, torch.Tensor):
        if bmm1_scale.dtype != torch.float32:
            raise TypeError("bmm1_scale tensor must have dtype torch.float32")
        bmm1_scale = bmm1_scale * log2e
    if isinstance(bmm2_scale, torch.Tensor):
        if bmm2_scale.dtype != torch.float32:
            raise TypeError("bmm2_scale tensor must have dtype torch.float32")

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
    run_func(
        out,
        query_flat,
        primary_kv_cache,
        swa_kv_cache,
        workspace_buffer,
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
            The implementation backend, could be ``auto``/``fa2`` or ``fa3``. Defaults to ``auto``.
            If set to ``auto``, the function will automatically choose the backend based on the
            device architecture and kernel availability. If ``cutlass`` is provided, the MLA
            kernels will be generated by CUTLASS and only float_workspace_buffer is required and
            other arguments are ignored.
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
        """
        if self._backend == "cutlass":
            if return_lse:
                raise ValueError("return_lse does not support cutlass backend for now.")
            if profiler_buffer is not None:
                raise ValueError(
                    "profiler_buffer does not support cutlass backend for now."
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
            *profiler_args,
        )

        return (out, lse) if return_lse else out


# ---------------------------------------------------------------------------
# Autotuning support for trtllm_batch_decode_with_kv_cache_mla
# ---------------------------------------------------------------------------

# Trtllm-gen kernel has a hardcoded max_batch_size = 8192 cap in
# csrc/trtllm_fmha_kernel_launcher.cu:200 (the counter-region semaphore array
# is sized for this max). Profiling beyond this would alias semaphores and
# produce non-representative measurements.
_TRTLLM_GEN_MLA_MAX_BATCH = 8192

# Size of the trtllm-gen workspace counter region (multi-block semaphores)
# per csrc/trtllm_fmha_kernel_launcher.cu:200: max_batch_size * max_num_qo_heads
# * sizeof(uint32_t) = 8192 * 256 * 4 = 8 MB. trtllm-gen places this counter
# slab at the head of the workspace_buffer and self-resets it at the end of
# every launch, so back-to-back trtllm-gen launches keep it valid without any
# host-side zeroing.
_TRTLLM_GEN_MLA_COUNTER_REGION_BYTES = 8192 * 256 * 4


def _cute_dsl_workspace_view(workspace_buffer: torch.Tensor) -> torch.Tensor:
    """Sub-view of the shared workspace that skips trtllm-gen's counter region.

    cute-dsl carves its scratch from offset 0 of whatever buffer it is given;
    offsetting past the 8 MB counter region keeps it from writing into the
    bytes trtllm-gen needs zero on entry. Costs 8 MB of usable workspace
    (callers should size the buffer accordingly; the recommended 128 MB has
    ample headroom).
    """
    workspace_i8 = workspace_buffer.reshape(-1).view(torch.int8)
    return workspace_i8[_TRTLLM_GEN_MLA_COUNTER_REGION_BYTES:]


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

        # cute-dsl gives up the counter region only when trtllm-gen shares the
        # buffer, so its usable size excludes that reservation only then.
        reserved = (
            _TRTLLM_GEN_MLA_COUNTER_REGION_BYTES if "trtllm-gen" in runner_names else 0
        )
        cute_dsl_cap = _cute_dsl_max_supported_batch(
            workspace_bytes=workspace_buffer.numel() * workspace_buffer.element_size()
            - reserved,
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
    from ..autotuner import DynamicTensorSpec, TuningConfig
    from ..fused_moe.utils import make_bucket_mapper

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

        self._run(
            out,
            None,  # fp4 output (unsupported by wrapper)
            query_flat,
            self.kv_cache,
            self.kv_cache,  # kv passed twice (K/V views over the same buffer)
            self.workspace_buffer,
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
        reserve_counter_region: bool = False,
    ):
        from ..cute_dsl.attention import cute_dsl_mla_decode

        self._run = cute_dsl_mla_decode
        self.kv_cache = kv_cache
        # Only skip trtllm-gen's counter region when trtllm-gen shares this
        # buffer (the "auto" path); a standalone cute-dsl runner owns the whole
        # buffer and can use it from offset 0.
        self.workspace_buffer = (
            _cute_dsl_workspace_view(workspace_buffer)
            if reserve_counter_region
            else workspace_buffer
        )
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
    seq_lens: torch.Tensor,
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
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Parameters
    ----------
    query: [batch_size, q_len_per_request, num_heads, head_dim_qk], head_dim_qk = qk_nope_head_dim (kv_lora_rank) + qk_rope_head_dim, should be concated q_nope + q_rope; q_len_per_request is the MTP query length.
    kv_cache: [num_pages, page_size, head_dim_ckv + head_dim_kpe] or [num_pages, 1, page_size, head_dim_ckv + head_dim_kpe], should be concated ckv_cache + kpe_cache. Both 3D and 4D formats are supported for backward compatibility.
    workspace_buffer: [num_semaphores, 4], used for multi_block mode. Must be initialized to 0 for its first use.
    qk_nope_head_dim: qk_nope_head_dim, must be 128 or 64
    kv_lora_rank: kv_lora_rank, must be 512 or 256
    qk_rope_head_dim: qk_rope_head_dim, must be 64
    sparse_mla_top_k: sparse MLA top k, must be 0 for non-sparse MLA.
    block_tables: page table of kv cache.
        When ``uses_shared_paged_kv_idx`` is True (default): shape ``[batch_size, max_num_pages_per_seq]``.
        When ``uses_shared_paged_kv_idx`` is False: shape ``[batch_size, 2, max_num_pages_per_seq]``
        where dim 1 distinguishes K (0) and V (1) page indices. For MLA both rows will
        typically be identical since K and V share the same compressed representation.
    seq_lens: query_len
    max_seq_len: max sequence length for kv_cache
    out: output tensor, if not provided, will be allocated internally
    bmm1_scale: fused scale for mla bmm1 input.
        When using ``trtllm-gen`` backend, it can be a ``torch.Tensor`` with dtype ``torch.float32``.
        When using ``cute-dsl`` backend, only ``float`` values are supported.
    bmm2_scale: fused scale for mla bmm2 input.
        When using ``trtllm-gen`` backend, it can be a ``torch.Tensor`` with dtype ``torch.float32``.
        When using ``cute-dsl`` backend, only ``float`` values are supported.
    sinks: additional value per head in the denominator of the softmax.
        Supported by all three backends.  On ``cute-dsl`` this requires
        the modular implementation; ``cute_dsl_impl="auto"`` (the default)
        promotes to modular automatically, and ``cute_dsl_impl="monolithic"``
        with sinks set raises :class:`ValueError`.
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
        The implementation backend, could be ``auto``/``xqa``, ``trtllm-gen``, or ``cute-dsl``. Defaults to ``auto``.
        When set to ``auto``, the backend will be chosen based on the device architecture and kernel availability.
        For sm_100 and sm_103 (blackwell architecture), ``auto`` will choose ``trtllm-gen`` backend.
        For sm_120 (blackwell architecture), ``auto`` will choose ``xqa`` backend.
        The ``cute-dsl`` backend has two interchangeable implementations
        (``monolithic`` and ``modular``) on the same shape/dtype envelope;
        which one runs is controlled by the ``cute_dsl_impl`` kwarg below.
    is_var_seq : bool
        Whether the sequence length is variable.
        If True, the sequence length is variable.
        Otherwise,the sequence length is fixed for all the requests in the batch.
    uses_shared_paged_kv_idx : bool = True
        Whether the K and V page indices are shared as a unified index.
        True (default) uses vLLM/FlashInfer layout with a 2D page table.
        False uses TRT-LLM layout with a 3D page table ``[batch_size, 2, max_num_pages_per_seq]``.
        False is only supported for trtllm-gen backend.
    lse : Optional[torch.Tensor] = None
        Optional pre-allocated buffer for Log-Sum-Exp values. Supported by
        ``trtllm-gen`` and ``cute-dsl`` backends. Must have dtype
        ``torch.float32``. Accepted shapes:

        * ``[batch_size * q_len_per_request, num_qo_heads]`` (trtllm-gen
          native; accepted by both backends), or
        * ``[batch_size, q_len_per_request, num_qo_heads]`` (cute-dsl native;
          also accepted by cute-dsl).

        If ``return_lse`` is True and this is None, a buffer will be
        allocated by the backend.
    return_lse : bool = False
        Whether to return LSE values. Supported by ``trtllm-gen`` and
        ``cute-dsl`` backends. When True, the function returns ``(out, lse)``.
    cute_dsl_impl : str = "auto"
        Which cute-dsl implementation to use.  Honored only when
        ``backend="cute-dsl"``; ignored for other backends.

        * ``"auto"`` (default) — picks monolithic by default, automatically
          promoted to modular when the call uses a feature monolithic
          doesn't support (currently ``sinks``).
        * ``"modular"`` — strict.  Always run the modular kernels.
        * ``"monolithic"`` — strict.  Always run the monolithic kernels;
          raise :class:`ValueError` if the call uses any modular-only
          feature (e.g. ``sinks``).

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
    When called under ``flashinfer.autotune(True)`` with ``backend="auto"``, this
    function profiles both ``trtllm-gen`` and ``cute-dsl`` across a bucketed batch
    sweep up to each runner's kernel/workspace cap and caches the winning runner
    per shape signature. Subsequent calls under ``autotune(False)`` dispatch to
    the cached choice; any batch outside the tuned range falls back to a default
    runner with a one-time warning.

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

    if backend == "auto" and get_compute_capability(query.device)[0] != 10:
        backend = "xqa"

    if backend == "xqa":
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

    if backend not in ("auto", "trtllm-gen", "cute-dsl"):
        raise ValueError(f"Backend {backend} not supported")

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

    if skip_softmax_threshold_scale_factor is not None and sparse_mla_top_k != 0:
        raise ValueError("skip_softmax is not supported for sparse MLA")

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
    )
    if backend == "cute-dsl":
        if cute_dsl_reason is not None:
            raise ValueError(cute_dsl_reason)
        runner_names = ["cute-dsl"]
    elif backend == "trtllm-gen":
        runner_names = ["trtllm-gen"]
    else:  # backend == "auto"
        runner_names = ["trtllm-gen"]
        if cute_dsl_reason is None:
            runner_names.append("cute-dsl")

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
                # Reserve trtllm-gen's counter region only when it co-runs on
                # the shared workspace (the "auto" path).
                reserve_counter_region="trtllm-gen" in runner_names,
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
        Pre-allocated workspace buffer.  Must be zero-initialized on first use.
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
