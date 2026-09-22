# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Optional cuDNN DSA implementation of token-sparse, absorbed MLA."""

import functools
from numbers import Real
from typing import Any, Optional

import torch

from ..utils import check_shape_dtype_device, get_compute_capability, log2e


# H64 DSA has one CTA per query row. Keep the decode implementation for small
# batches; prefer it from 128 rows based on the measured SM100 crossover.
_AUTO_MIN_QUERY_ROWS = 128
_operations: dict[tuple, Any] = {}


@functools.cache
def _get_sparse_attention_forward():
    # cuDNN is optional. Missing/older installations must not break auto or
    # import FlashInfer's dense cuDNN graph backend on this path.
    try:
        import cudnn
        from packaging.version import Version

        if Version(cudnn.__version__) < Version("1.29.0"):
            return None
        from cudnn import DSA

        return getattr(DSA, "SparseAttentionForward", None)
    except (ImportError, OSError):
        return None


def _incompatibility_reason(
    query,
    kv_cache,
    *,
    kv_lora_rank,
    qk_rope_head_dim,
    sparse_mla_top_k,
    bmm1_scale,
    bmm2_scale,
    sinks,
    skip_softmax_threshold_scale_factor,
    enable_pdl,
    uses_shared_paged_kv_idx,
    multi_ctas_kv_counter_buffer,
    use_fp16_softmax,
) -> Optional[str]:
    if not query.is_cuda or get_compute_capability(query.device) not in (
        (10, 0),
        (10, 3),
    ):
        return "requires an SM100 or SM103 GPU"
    if query.ndim not in (3, 4) or query.shape[-2] != 64:
        return "requires 64 query heads in a 3D or 4D query tensor"
    if query.numel() == 0:
        return "requires at least one query row"
    if (
        kv_lora_rank != 512
        or qk_rope_head_dim not in (0, 64)
        or query.shape[-1] != 512 + qk_rope_head_dim
    ):
        return "requires kv_lora_rank=512 and QK dimension 512 or 576"
    if query.dtype != torch.bfloat16 or kv_cache.dtype != torch.bfloat16:
        return "requires BF16 queries and KV cache"
    if (
        kv_cache.ndim not in (3, 4)
        or (kv_cache.ndim == 4 and kv_cache.shape[1] != 1)
        or kv_cache.shape[-1] != query.shape[-1]
    ):
        return "requires packed KV [pages, page_size, D] or [pages, 1, page_size, D]"
    if (
        not query.is_contiguous()
        or not kv_cache.is_contiguous()
        or query.data_ptr() % 16
        or kv_cache.data_ptr() % 16
    ):
        return "requires contiguous, 16-byte-aligned queries and KV cache"
    if sparse_mla_top_k <= 0:
        return "requires sparse_mla_top_k > 0"
    if not isinstance(bmm1_scale, Real) or not isinstance(bmm2_scale, Real):
        return "requires scalar bmm1_scale and bmm2_scale"
    if bmm2_scale != 1.0:
        return "requires bmm2_scale=1.0"
    if sinks is not None:
        return "does not support sinks"
    if skip_softmax_threshold_scale_factor is not None:
        return "does not support skip_softmax"
    if enable_pdl:
        return "does not support enable_pdl=True"
    if not uses_shared_paged_kv_idx:
        return "requires uses_shared_paged_kv_idx=True"
    if multi_ctas_kv_counter_buffer is not None:
        return "does not use multi_ctas_kv_counter_buffer"
    if use_fp16_softmax:
        return "does not support use_fp16_softmax"
    return None


def try_cudnn_sparse_mla(
    *,
    query,
    kv_cache,
    workspace_buffer,
    kv_lora_rank,
    qk_rope_head_dim,
    block_tables,
    seq_lens,
    sparse_mla_top_k,
    out,
    bmm1_scale,
    bmm2_scale,
    sinks,
    skip_softmax_threshold_scale_factor,
    enable_pdl,
    uses_shared_paged_kv_idx,
    lse,
    return_lse,
    return_lse_base,
    cum_seq_lens_q,
    max_q_len,
    multi_ctas_kv_counter_buffer,
    sparse_mla_top_k_lens,
    use_fp16_softmax,
    required=False,
):
    """Return the cuDNN result, or None when auto should retain another backend.

    Only capability/dependency misses fall back. Invalid tensor metadata and
    execution failures are surfaced, rather than hidden by a second launch.
    """
    rows = (
        query.shape[0] * (query.shape[1] if query.ndim == 4 else 1)
        if query.ndim in (3, 4)
        else 0
    )
    if not required and (rows < _AUTO_MIN_QUERY_ROWS or sparse_mla_top_k <= 0):
        return None
    reason = _incompatibility_reason(
        query,
        kv_cache,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        sparse_mla_top_k=sparse_mla_top_k,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        sinks=sinks,
        skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
        enable_pdl=enable_pdl,
        uses_shared_paged_kv_idx=uses_shared_paged_kv_idx,
        multi_ctas_kv_counter_buffer=multi_ctas_kv_counter_buffer,
        use_fp16_softmax=use_fp16_softmax,
    )
    if reason is not None:
        if required:
            raise ValueError(f"cudnn sparse MLA {reason}")
        return None
    # These storage restrictions are specific to cuDNN; auto must retain the
    # existing backend when a caller supplies otherwise valid strided tensors
    # or a workspace sized for that backend. Do not allocate or copy to qualify.
    stat_bytes = rows * 64 * 4
    needed_bytes = stat_bytes * (1 if lse is not None or return_lse else 2)
    storage_reason = None
    if not block_tables.is_contiguous():
        storage_reason = "requires contiguous sparse block_tables"
    elif (
        sparse_mla_top_k_lens is not None and not sparse_mla_top_k_lens.is_contiguous()
    ):
        storage_reason = "requires contiguous sparse_mla_top_k_lens"
    elif out is not None and (not out.is_contiguous() or out.data_ptr() % 16):
        storage_reason = "requires contiguous, 16-byte-aligned out"
    elif lse is not None and (not lse.is_contiguous() or lse.data_ptr() % 4):
        storage_reason = "requires contiguous, 4-byte-aligned lse"
    elif (
        workspace_buffer.device != query.device
        or not workspace_buffer.is_contiguous()
        or workspace_buffer.data_ptr() % 4
        or workspace_buffer.numel() * workspace_buffer.element_size() < needed_bytes
    ):
        storage_reason = (
            "requires a contiguous, 4-byte-aligned workspace on the query device "
            f"with at least {needed_bytes} bytes"
        )
    if storage_reason is not None:
        if required:
            raise ValueError(f"cudnn sparse MLA {storage_reason}")
        return None
    operation_type = _get_sparse_attention_forward()
    if operation_type is None:
        if required:
            raise ImportError(
                "cudnn sparse MLA requires nvidia-cudnn-frontend>=1.29.0 "
                "with CuTe DSL support; install nvidia-cudnn-frontend[cutedsl]"
            )
        return None

    device = query.device
    with torch.cuda.device(device):
        capturing = torch.cuda.is_current_stream_capturing()
    if kv_cache.device != device:
        raise ValueError("KV cache must be on the query device")
    if cum_seq_lens_q is None:
        if query.ndim != 4:
            raise ValueError("3D query requires cum_seq_lens_q")
        batch_size = query.shape[0]
    else:
        if query.ndim != 3:
            raise ValueError("cum_seq_lens_q requires a 3D query")
        check_shape_dtype_device(
            cum_seq_lens_q, None, torch.int32, device, "cum_seq_lens_q"
        )
        if cum_seq_lens_q.ndim != 1 or cum_seq_lens_q.numel() < 2:
            raise ValueError("cum_seq_lens_q must be 1D with at least two entries")
        batch_size = cum_seq_lens_q.numel() - 1
        if max_q_len is None:
            if capturing:
                raise ValueError(
                    "Provide max_q_len for compact queries during CUDA graph capture"
                )
            offsets = cum_seq_lens_q.cpu()
            lengths = offsets[1:] - offsets[:-1]
            if offsets[0].item() != 0 or offsets[-1].item() != rows:
                raise ValueError(
                    "cum_seq_lens_q must start at 0 and end at the query row count"
                )
            if (lengths < 0).any().item():
                raise ValueError("cum_seq_lens_q must be monotonically non-decreasing")
        elif max_q_len <= 0:
            raise ValueError("max_q_len must be greater than 0")
    if seq_lens is not None:
        check_shape_dtype_device(
            seq_lens, (batch_size,), torch.int32, device, "seq_lens"
        )
    check_shape_dtype_device(
        block_tables,
        query.shape[:-2] + (sparse_mla_top_k,),
        torch.int32,
        device,
        "block_tables",
    )
    if sparse_mla_top_k_lens is not None:
        check_shape_dtype_device(
            sparse_mla_top_k_lens, (rows,), torch.int32, device, "sparse_mla_top_k_lens"
        )

    # D576 TRTLLM bounds the selected-list prefix by each query's causal KV
    # length, even when later slots contain valid physical IDs. D512 instead
    # uses explicit per-query top-k lengths, independent of seq_lens.
    topk_length = sparse_mla_top_k_lens
    if topk_length is None and qk_rope_head_dim == 64 and seq_lens is not None:
        if cum_seq_lens_q is None:
            q_len = query.shape[1]
            if q_len == 1:
                topk_length = seq_lens.contiguous()
            else:
                positions = torch.arange(q_len, device=device, dtype=torch.int32)
                topk_length = (
                    (seq_lens[:, None] - q_len + positions[None, :] + 1)
                    .reshape(rows)
                    .clamp_min_(0)
                )
        else:
            positions = torch.arange(rows, device=device, dtype=torch.int32)
            requests = torch.searchsorted(cum_seq_lens_q[1:], positions, right=True)
            topk_length = (
                seq_lens[requests] - cum_seq_lens_q[requests + 1] + positions + 1
            ).clamp_min_(0)

    shape = query.shape[:-1] + (512,)
    if out is None:
        out = torch.empty(shape, dtype=torch.bfloat16, device=device)
    else:
        check_shape_dtype_device(out, shape, torch.bfloat16, device, "out")
    user_lse = lse
    if lse is not None:
        check_shape_dtype_device(lse, None, torch.float32, device, "lse")
        if lse.shape not in ((rows, 64), query.shape[:-1]):
            raise ValueError("lse must have shape [total_q, H] or [batch, q_len, H]")
    elif return_lse:
        user_lse = lse = torch.empty((rows, 64), dtype=torch.float32, device=device)

    # Scratch belongs to the caller, not the cached operation. Distinct streams
    # and captured graphs can execute concurrently with independent workspaces.
    scratch = workspace_buffer.view(torch.uint8).view(-1)
    max_logits = scratch[:stat_bytes].view(torch.float32).view(rows, 64)
    kernel_lse = (
        scratch[stat_bytes : 2 * stat_bytes].view(torch.float32).view(rows, 64)
        if lse is None
        else lse.view(rows, 64)
    )
    q = query.view(rows, 64, query.shape[-1])
    kv = kv_cache.view(-1, query.shape[-1])
    indices = block_tables.view(rows, sparse_mla_top_k)
    key = (device, query.shape[-1], sparse_mla_top_k, topk_length is not None)
    op = _operations.get(key)
    if op is None:
        if capturing:
            raise RuntimeError(
                "Warm up cudnn sparse MLA outside CUDA graph capture before using this configuration"
            )
        op = operation_type(
            q, kv, indices, sample_topk_length=topk_length, indexer_topk=0
        )
        op.check_support()
        op.compile()
    op.execute(
        q,
        kv,
        indices,
        topk_length=topk_length,
        softmax_scale=float(bmm1_scale),
        out=out.view(rows, 64, 512),
        max_logits=max_logits,
        lse=kernel_lse,
    )
    # compile() only performs APIBase setup; execute() warms the concrete CuTe
    # kernel. Cache the immutable operation only after that launch succeeds.
    _operations[key] = op
    # Auto previously routed compact queries requesting LSE to monolithic
    # CuTeDSL (natural log), and fixed queries to TRTLLM-GEN (base 2).
    natural_lse = return_lse_base == "basee" or (
        return_lse_base is None and not required and cum_seq_lens_q is not None
    )
    if lse is not None and not natural_lse:
        kernel_lse.mul_(log2e)
    return (out, user_lse) if return_lse else out
