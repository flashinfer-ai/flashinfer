import functools
from typing import Optional

import torch

from ..jit import get_cudnn_fmha_gen_module


def cudnn_batch_prefill_with_kv_cache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    scale: float,
    workspace_buffer: torch.Tensor,
    actual_seq_lens_q: torch.Tensor,
    actual_seq_lens_kv: torch.Tensor,
    block_tables: torch.Tensor,
    num_pages_per_seq: int,
    causal: bool,
    return_lse: bool,
    batch_offsets: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Performs batched prefill attention with paged KV cache using cuDNN.

    Args:
        q: Query tensor of shape (batch_size, seq_len_q, num_heads_qo, head_dim), seq_len_q is the maximum sequence length of queries in the batch
        k_cache: Key cache tensor of shape   (total_num_pages, num_heads_kv, page_size, head_dim)
        v_cache: Value cache tensor of shape (total_num_pages, num_heads_kv, page_size, head_dim)
        scale: Scaling factor for attention scores, typically 1/sqrt(head_dim)
        workspace_buffer: Workspace buffer for cuDNN operations. Scales with batch size. 128 MB should be sufficient for most cases
        actual_seq_lens_q: Actual sequence lengths for queries per batch, shape (batch_size,) on CPU,
        actual_seq_lens_kv: Actual sequence lengths for key/values per batch, shape (batch_size,) on CPU
        block_tables: Page table mapping for KV cache, shape (batch_size, num_pages_per_seq) on GPU
        num_pages_per_seq: Number of pages allocated per sequence (s_kv / page_size)
        causal: Whether to apply causal masking (must be True)
        return_lse: Whether to return log-sum-exp values
        out: Optional pre-allocated output tensor
        lse: Optional pre-allocated tensor for log-sum-exp values if return_lse is True else returns None

    Returns:
        Output tensor of shape (batch_size, seq_len_q, num_heads_qo, head_dim)
        If return_lse is True, also returns log-sum-exp tensor of shape (batch_size, seq_len_q, num_heads_qo)

    Note:
        Currently only supports causal attention (causal must be True)
        All tensors must be contiguous and on the same CUDA device
        Query and KV heads can have different sizes (num_heads_qo >= num_heads_kv)
    """

    bs = q.shape[0]
    s_q = q.shape[1]
    h_qo = q.shape[2]
    d_vo = v_cache.shape[3]

    if return_lse:
        if lse is None:
            lse = torch.empty(bs, s_q, h_qo, device=q.device, dtype=torch.float32)
        if lse.shape != (bs, s_q, h_qo):
            raise ValueError("lse must have shape (bs, s_q, h_qo)")

    if out is None:
        # print(f"out is nan, creating new out tensor")
        # out = torch.full((bs, s_q, h_qo, d_vo), float('3.0'), device=q.device, dtype=q.dtype)
        out = torch.randn((bs, h_qo, s_q, d_vo), device=q.device, dtype=q.dtype)
        out = out.as_strided((bs, h_qo, s_q, d_vo), (h_qo * d_vo, d_vo, h_qo * d_vo, 1))

    actual_seq_lens_q_gpu = actual_seq_lens_q.to(q.device)
    actual_seq_lens_kv_gpu = actual_seq_lens_kv.to(q.device)

    run_func = get_cudnn_fmha_gen_module().prefill
    run_func(
        q,
        k_cache,
        v_cache,
        scale,
        workspace_buffer,
        actual_seq_lens_q,
        actual_seq_lens_kv,
        actual_seq_lens_q_gpu,
        actual_seq_lens_kv_gpu,
        block_tables,
        num_pages_per_seq,
        causal,
        return_lse,
        out,
        lse,
        batch_offsets,
    )

    # import csv
    # import numpy as np
    # if return_lse:
    #     # Save lse tensor to file
    #     # Convert LSE tensor to numpy array and save to file with descriptive name
    #     lse_numpy = lse.cpu().numpy()
    #     with open('cudnn_prefill_lse.csv', 'w', newline='') as f:
    #         csv.writer(f).writerows([[float(x)] for x in lse_numpy.flatten()])
    return out
