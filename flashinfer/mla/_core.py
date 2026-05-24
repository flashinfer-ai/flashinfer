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
from ..trace.templates.attention import (
    mla_paged_decode_trace,
    trtllm_batch_decode_mla_trace,
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
    if num_heads not in (64, 128):
        raise ValueError(f"Expected 64 or 128 query heads, got {num_heads}")

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
        qo_indptr_buf : Optional[torch.Tensor]
            The user reserved buffer to store the ``qo_indptr`` array, the size of the buffer
            should be ``[batch_size + 1]``.
            This argument is only effective when ``use_cuda_graph`` is ``True``.
        kv_indptr_buf : Optional[torch.Tensor]
            The user reserved buffer to store the ``kv_indptr`` array, the size of the buffer
            should be ``[batch_size + 1]``.
            This argument is only effective when ``use_cuda_graph`` is ``True``.
        kv_indices_buf : Optional[torch.Tensor]
            The user reserved buffer to store the ``kv_indices`` array.
            This argument is only effective when ``use_cuda_graph`` is ``True``.
        kv_len_arr_buf : Optional[torch.Tensor]
            The user reserved buffer to store the ``kv_len_arr`` array, the size of the buffer
            should be ``[batch_size]``.
            This argument is only effective when ``use_cuda_graph`` is ``True``.
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


@flashinfer_api(trace=trtllm_batch_decode_mla_trace)
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
        Optional pre-allocated buffer for Log-Sum-Exp values. Only supported by
        ``trtllm-gen`` backend. Must have shape
        ``[batch_size * q_len_per_request, num_qo_heads]`` with dtype
        ``torch.float32``. If ``return_lse`` is True and this is None, a buffer
        will be allocated.
    return_lse : bool = False
        Whether to return LSE values. Only supported by ``trtllm-gen`` backend.
        When True, the function returns ``(out, lse)``.
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
    """
    if backend == "auto":
        backend = (
            "trtllm-gen" if get_compute_capability(query.device)[0] == 10 else "xqa"
        )
    if isinstance(bmm1_scale, torch.Tensor):
        if bmm1_scale.dtype != torch.float32:
            raise TypeError("bmm1_scale tensor must have dtype torch.float32")
        bmm1_scale = bmm1_scale * log2e
    if isinstance(bmm2_scale, torch.Tensor):
        if bmm2_scale.dtype != torch.float32:
            raise TypeError("bmm2_scale tensor must have dtype torch.float32")
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
    elif backend == "trtllm-gen":
        enable_pdl = (
            device_support_pdl(query.device) if enable_pdl is None else enable_pdl
        )
        run_func = get_trtllm_gen_fmha_module().trtllm_paged_attention_decode
        sm_count = get_device_sm_count(query.device)

        # Extract block_size (works for both 3D and 4D)
        block_size = kv_cache.size(-2)
        if (
            block_size != 32 and block_size != 64
        ):  # todo(Yingyi): add support for more block sizes?
            raise ValueError(f"Supported block_size are 32 and 64, got {block_size}")

        if skip_softmax_threshold_scale_factor is not None and sparse_mla_top_k != 0:
            raise ValueError("skip_softmax is not supported for sparse MLA")

        # Validate and normalize to 4D
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

        batch_size = query.size(0)
        max_q_len = query.size(1)
        num_qo_heads = query.size(2)
        query = query.flatten(0, 1)  # [B*S, H, D]

        if return_lse:
            lse_shape = (batch_size * max_q_len, num_qo_heads)
            if lse is None:
                lse = torch.empty(lse_shape, dtype=torch.float32, device=query.device)
            else:
                check_shape_dtype_device(
                    lse, lse_shape, torch.float32, query.device, "lse"
                )
            lse_stride_tokens = lse.stride(0)
            lse_stride_heads = lse.stride(1)
        else:
            lse = None
            lse_stride_tokens = 0
            lse_stride_heads = 0

        run_func(
            out,
            None,  # fp4 output not supported in wrapper api yet.
            query,
            kv_cache,
            kv_cache,
            workspace_buffer,
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
            None,  # cum_seq_lens_q
            None,  # key_block_scales
            None,  # value_block_scales
            skip_softmax_threshold_scale_factor,
            uses_shared_paged_kv_idx,
            lse,
            lse_stride_tokens,
            lse_stride_heads,
        )

        if return_lse:
            return out, lse
        return out
    elif backend == "cute-dsl":
        enable_pdl = (
            device_support_pdl(query.device) if enable_pdl is None else enable_pdl
        )
        cc = get_compute_capability(query.device)
        if cc[0] < 10:
            raise RuntimeError(
                f"cute-dsl backend (MLA decode kernel) requires SM100+, got SM{cc[0]}{cc[1]}"
            )
        from flashinfer.cute_dsl.attention import cute_dsl_mla_decode

        if isinstance(bmm1_scale, torch.Tensor):
            raise ValueError(
                "cute-dsl backend (MLA decode kernel) does not support tensor bmm1_scale, "
                "please pass a float value"
            )
        if isinstance(bmm2_scale, torch.Tensor):
            raise ValueError(
                "cute-dsl backend (MLA decode kernel) does not support tensor bmm2_scale, "
                "please pass a float value"
            )
        # `sinks` is supported via the modular AttentionWithSink variant; the
        # dispatcher in flashinfer.cute_dsl.attention.mla_dispatch will force
        # impl="modular" when sinks is set (monolithic has no variant path).
        # The public sinks signature is Optional[List[torch.Tensor]] for
        # legacy reasons, but every backend (xqa, trtllm-gen, cute-dsl)
        # treats it as a single per-head tensor.  Normalise here so the
        # downstream cute-dsl path sees a tensor or None; reject the
        # ambiguous len>1 case loudly rather than silently dropping tail
        # entries.
        cute_dsl_sinks: Optional[torch.Tensor] = None
        if sinks is not None:
            if isinstance(sinks, (list, tuple)):
                if len(sinks) != 1:
                    raise ValueError(
                        f"cute-dsl backend (MLA decode kernel) expects sinks "
                        f"to be a single tensor or a length-1 list/tuple; got "
                        f"len={len(sinks)}."
                    )
                cute_dsl_sinks = sinks[0]
            else:
                cute_dsl_sinks = sinks
        if sparse_mla_top_k > 0:
            raise ValueError(
                "cute-dsl backend (MLA decode kernel) does not support sparse_mla_top_k"
            )
        if skip_softmax_threshold_scale_factor is not None:
            raise ValueError(
                "cute-dsl backend (MLA decode kernel) does not support skip_softmax_threshold_scale_factor"
            )
        if not uses_shared_paged_kv_idx:
            raise ValueError(
                "cute-dsl backend (MLA decode kernel) does not support separate KV page indices "
                "(uses_shared_paged_kv_idx=False)"
            )
        if return_lse or lse is not None:
            raise NotImplementedError(
                "cute-dsl backend (MLA decode kernel) does not support return_lse/lse output"
            )

        return cute_dsl_mla_decode(
            query=query,
            kv_cache=kv_cache,
            workspace_buffer=workspace_buffer,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            softmax_scale=bmm1_scale,
            output_scale=bmm2_scale,
            out=out,
            is_var_seq=is_var_seq,
            enable_pdl=enable_pdl,
            sinks=cute_dsl_sinks,
            cute_dsl_impl=cute_dsl_impl,
        )
    else:
        raise ValueError(f"Backend {backend} not supported")


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
    """
    Parameters:
    query: [batch_size, q_len_per_request, num_heads, head_dim_qk], head_dim_qk = qk_nope_head_dim (kv_lora_rank) + qk_rope_head_dim, should be concated q_nope + q_rope; q_len_per_request is the MTP query length.
    kv_cache: [num_pages, page_size, head_dim_ckv + head_dim_kpe] or [num_pages, 1, page_size, head_dim_ckv + head_dim_kpe], should be concated ckv_cache + kpe_cache. Both 3D and 4D formats are supported for backward compatibility.
    workspace_buffer: torch.Tensor. Must be initialized to 0 for its first use.
    qk_nope_head_dim: qk_nope_head_dim, must be 128
    kv_lora_rank: kv_lora_rank, must be 512
    qk_rope_head_dim: qk_rope_head_dim, must be 64
    block_tables: page_table of kv cache, [batch_size, num_pages]
    seq_lens: query_len
    max_seq_len: max sequence length for kv_cache
    out: output tensor, if not provided, will be allocated internally
    bmm1_scale: fused scale for mla bmm1 input. Can be a float or a torch.Tensor.
    bmm2_scale: fused scale for mla bmm2 input. Can be a float or a torch.Tensor.
    sinks: additional value per head in the denominator of the softmax.

    Note:
    In MLA, the actual BMM1 and BMM2 scales applied would be fused as:
    bmm1_scale = q_scale * k_scale * sm_scale / (head_dim_qk ** 0.5)
    bmm2_scale = v_scale * o_scale

    The two scale factors should be static constant for cuda graph capture.
    Either (bmm1_scale, bmm2_scale) or (bmm1_scale_log2_tensor, bmm2_scale_tensor) should be provided.

    For static constant scale factors, the scale factors should be provided as float.
        - (bmm1_scale, bmm2_scale)
    For on-device fused scale tensors, which could dynamically change, the scale factors should be provided as torch.Tensor.
        - (bmm1_scale_log2_tensor, bmm2_scale_tensor)
        - Currently, only fp8 tensor core operation supports this mode.
    When both are provided, the dynamic scale factor tensors will be used.
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
