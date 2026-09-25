# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""MiniMax-M3's per-token speculative decode contract on SM100/SM103."""

import functools
import math

import torch

from ..api_logging import flashinfer_api
from ..decode import trtllm_batch_decode_with_kv_cache
from ..utils import (
    get_compute_capability,
    get_device_sm_count,
    get_trtllm_gen_multi_ctas_kv_counter_bytes,
)
from ..trace.templates.minimax_m3 import minimax_m3_sparse_attn_decode_trace_dispatch


@functools.cache
def _get_metadata_module():
    from ..jit.minimax_m3 import gen_minimax_m3_module

    return gen_minimax_m3_module().build_and_load()


class MiniMaxM3SparseDecodeWorkspace:
    """Reusable storage for :func:`minimax_m3_sparse_attn_decode`.

    Allocate once outside CUDA graph capture, then warm the complete attention
    call before capture. A workspace must not be used by concurrent streams or
    overlapping graph replays. Page tables, sequence lengths, selections, and
    the scalar K/V scales may change in place between replays. The output is
    caller-owned and never allocated by the attention call.

    Parameters
    ----------
    batch_size, num_qo_heads, num_kv_heads, decode_query_len : int
        Fixed batch geometry. GQA group size must be 16, and query length 1–8.
        TP1 uses 64/4 heads; one TP4 rank uses 16/1 heads.
    device : torch.device or str
        An SM100/SM103 CUDA device.
    workspace_buffer : torch.Tensor, optional
        Contiguous uint8 attention scratch buffer. If omitted, allocate 128 MiB
        once. This scratch belongs to TRT-LLM, not a materialized/gathered KV cache.
    """

    def __init__(
        self,
        batch_size: int,
        num_qo_heads: int,
        num_kv_heads: int,
        decode_query_len: int,
        *,
        device: torch.device | str = "cuda",
        workspace_buffer: torch.Tensor | None = None,
    ):
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("MiniMax-M3 sparse decode requires a CUDA device")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if get_compute_capability(device) not in {(10, 0), (10, 3)}:
            raise ValueError("MiniMax-M3 sparse decode requires SM100 or SM103")
        if not 1 <= batch_size <= 256 or not 1 <= decode_query_len <= 8:
            raise ValueError(
                "batch_size must be in [1, 256] and decode_query_len in [1, 8]"
            )
        if num_kv_heads not in (1, 4) or num_qo_heads != num_kv_heads * 16:
            raise ValueError(
                "supported (num_qo_heads, num_kv_heads) are (16, 1) and (64, 4)"
            )
        self.device = device
        self.batch_size = batch_size
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.decode_query_len = decode_query_len
        self.total_q = batch_size * decode_query_len
        if workspace_buffer is None:
            workspace_buffer = torch.empty(
                128 * 1024 * 1024, dtype=torch.uint8, device=device
            )
        if (
            workspace_buffer.device != device
            or workspace_buffer.dtype != torch.uint8
            or not workspace_buffer.is_contiguous()
        ):
            raise ValueError(
                "workspace_buffer must be contiguous uint8 on the workspace device"
            )
        self.attention_buffer = workspace_buffer
        self.sparse_pages = torch.empty(
            (num_kv_heads, self.total_q, 16), dtype=torch.int32, device=device
        )
        self.sparse_lens = torch.empty(
            (num_kv_heads, self.total_q), dtype=torch.int32, device=device
        )
        self.qk_scale_log2 = torch.empty(1, dtype=torch.float32, device=device)
        self.counter_buffer = torch.zeros(
            get_trtllm_gen_multi_ctas_kv_counter_bytes(
                self.total_q, num_qo_heads, get_device_sm_count(device)
            ),
            dtype=torch.uint8,
            device=device,
        )


def _check_tensor(tensor, name, shape, dtype, device, *, contiguous=True):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if tuple(tensor.shape) != tuple(shape) or tensor.dtype != dtype:
        raise ValueError(f"{name} must have shape {tuple(shape)} and dtype {dtype}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}")
    if contiguous and not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous on {device}")


@flashinfer_api(trace=minimax_m3_sparse_attn_decode_trace_dispatch)
def minimax_m3_sparse_attn_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    out: torch.Tensor,
    workspace: MiniMaxM3SparseDecodeWorkspace,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Speculative sparse attention over the existing indexer's independent rows.

    This matches the request-major uniform-query-length decode convention of
    vLLM's ``minimax_m3_sparse_attn_decode``. For token ``i`` of request ``b``,
    the causal KV length is ``max(seq_lens[b] - decode_query_len + i + 1, 0)``.
    Only the first ``min(16, ceil(causal_length / 128))`` entries of its own
    selection are read. These must be distinct, valid logical page IDs; their
    order is arbitrary and unused tail entries are ignored. There is no
    cross-token sharing/union of selections and no automatically added tail page.

    Parameters
    ----------
    q : torch.Tensor
        Contiguous BF16 ``[batch_size * decode_query_len, Hq, 128]``.
    kv_cache : torch.Tensor
        Contiguous E4M3 ``[num_pages, Hkv, 128, 256]``. Each token holds K
        followed by V, with token stride 256. No repacking or KV gather occurs.
    topk_idx : torch.Tensor
        CUDA int32 ``[Hkv, total_q, 16]`` logical page selections. Strided
        views from a larger preallocated indexer output are supported directly.
    block_table : torch.Tensor
        CUDA int32 ``[batch_size, max_pages]`` (strided views allowed). Physical page IDs
        may be permuted and shared by requests (prefix caching).
    seq_lens : torch.Tensor
        CUDA int32 ``[batch_size]`` valid lengths after all queries (strided views allowed).
        Lengths may be ragged/non-page-aligned; maximum supported length 262144.
        Zero-length rows may be used for graph padding. Their outputs are
        unspecified (as in the Triton decode reference) and must be ignored.
    k_scale, v_scale : torch.Tensor
        CUDA float32 scalar dequantization scales (one element each). Their
        contents may change in place during CUDA graph replay. Per-token/head
        scale arrays are not supported by this TRT-LLM route.
    out : torch.Tensor
        Preallocated contiguous BF16 output with the same shape as ``q``.
    workspace : MiniMaxM3SparseDecodeWorkspace
        Preallocated storage specifying the fixed batch/head/query geometry.
        Warm the complete call eagerly before CUDA graph capture. All device
        allocations are outside this call, including the reduction counters.
    sm_scale : float, optional
        Finite positive host softmax scale, default ``128**-0.5``. K scale
        is applied in addition. Tensor-valued scales are rejected without
        reading their data.

    Returns
    -------
    torch.Tensor
        The caller's ``out`` tensor.

    Notes
    -----
    Metadata validation uses only host-side tensor descriptors, not tensor
    contents. Callers must supply valid indexer output and sequence lengths.
    Device work consists of sparse metadata preparation (including QK scale)
    and the complete native block-sparse attention call, with no Q quantization.
    """
    if isinstance(sm_scale, torch.Tensor):
        raise TypeError("sm_scale must be a host float, not a torch.Tensor")
    scale = 128**-0.5 if sm_scale is None else float(sm_scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("sm_scale must be finite and positive")
    if not isinstance(workspace, MiniMaxM3SparseDecodeWorkspace):
        raise TypeError("workspace must be a MiniMaxM3SparseDecodeWorkspace")
    w = workspace
    qshape = (w.total_q, w.num_qo_heads, 128)
    _check_tensor(q, "q", qshape, torch.bfloat16, w.device)
    _check_tensor(out, "out", qshape, torch.bfloat16, w.device)
    _check_tensor(
        topk_idx,
        "topk_idx",
        (w.num_kv_heads, w.total_q, 16),
        torch.int32,
        w.device,
        contiguous=False,
    )
    _check_tensor(
        seq_lens, "seq_lens", (w.batch_size,), torch.int32, w.device, contiguous=False
    )
    if (
        not isinstance(kv_cache, torch.Tensor)
        or kv_cache.ndim != 4
        or kv_cache.shape[0] <= 0
    ):
        raise ValueError("kv_cache must have shape [num_pages, Hkv, 128, 256]")
    _check_tensor(
        kv_cache,
        "kv_cache",
        (kv_cache.shape[0], w.num_kv_heads, 128, 256),
        torch.float8_e4m3fn,
        w.device,
    )
    if not isinstance(block_table, torch.Tensor) or block_table.ndim != 2:
        raise ValueError("block_table must have shape [batch_size, max_pages]")
    if not 1 <= block_table.shape[1] <= 2048:
        raise ValueError("block_table capacity must be in [1, 2048] pages")
    _check_tensor(
        block_table,
        "block_table",
        (w.batch_size, block_table.shape[1]),
        torch.int32,
        w.device,
        contiguous=False,
    )
    for tensor, name in ((k_scale, "k_scale"), (v_scale, "v_scale")):
        if not isinstance(tensor, torch.Tensor) or tensor.numel() != 1:
            raise ValueError(f"{name} must be a CUDA float32 scalar tensor")
        _check_tensor(tensor, name, tensor.shape, torch.float32, w.device)
    _get_metadata_module().prepare(
        topk_idx,
        block_table,
        seq_lens,
        k_scale,
        w.sparse_pages,
        w.sparse_lens,
        w.qk_scale_log2,
        w.decode_query_len,
        kv_cache.shape[0],
        scale,
    )
    # q_len_per_req=1 is essential: each flattened row has its own sparse
    # selection and causal length, even within one speculative request.
    return trtllm_batch_decode_with_kv_cache(
        q,
        (kv_cache[..., :128], kv_cache[..., 128:]),
        w.attention_buffer,
        w.sparse_pages,
        w.sparse_lens,
        16 * 128,
        bmm1_scale_log2=w.qk_scale_log2,
        bmm2_scale=v_scale,
        out=out,
        backend="trtllm-gen",
        q_len_per_req=1,
        enable_block_sparse_attention=True,
        multi_ctas_kv_counter_buffer=w.counter_buffer,
    )
