from typing import Optional

import torch

from ..jit import cudnn_fmha_gen_module


def get_cudnn_fmha_gen_module():
    return cudnn_fmha_gen_module()


def cudnn_batch_decode_with_kv_cache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    scale: float,
    workspace_buffer: torch.Tensor,
    actual_seq_lens_q: torch.Tensor,
    actual_seq_lens_kv: torch.Tensor,
    block_tables: torch.Tensor,
    num_pages_per_seq: int,
    batch_offsets: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Performs batched decode attention with paged KV cache using cuDNN.

    Args:
        q: Query tensor of shape (batch_size, seq_len_q, num_heads_qo, head_dim), seq_len_q is the maximum sequence length of queries in the batch
        k_cache: Key cache tensor of shape   (total_num_pages, num_heads_kv, page_size, head_dim)
        v_cache: Value cache tensor of shape (total_num_pages, num_heads_kv, page_size, head_dim)
        scale: Scaling factor for attention scores, typically 1/sqrt(head_dim)
        workspace_buffer: Workspace buffer for cuDNN operations. Scales with batch size. 128 MB should be sufficient for most cases
        actual_seq_lens_q: Actual sequence lengths for queries per batch, shape (batch_size,) on CPU
        actual_seq_lens_kv: Actual sequence lengths for key/values per batch, shape (batch_size,) on CPU
        block_tables: Page table mapping for KV cache, shape (batch_size, num_pages_per_seq)
        num_pages_per_seq: Number of pages allocated per sequence
        out: Optional pre-allocated output tensor

    Returns:
        Output tensor of shape (batch_size, seq_len_q, num_heads_qo, head_dim)

    Note:
        Currently only supports causal attention (causal must be True)
        All tensors must be contiguous and on the same CUDA device
        Query and KV heads can have different sizes (num_heads_qo >= num_heads_kv)
    """

    bs = q.shape[0]
    s_q = q.shape[1]
    h_qo = q.shape[2]
    d_vo = v_cache.shape[3]

    if out is None:
        out = torch.empty(bs, s_q, h_qo, d_vo, device=q.device, dtype=q.dtype)

    actual_seq_lens_q_gpu = actual_seq_lens_q.to(q.device)
    actual_seq_lens_kv_gpu = actual_seq_lens_kv.to(q.device)

    run_func = get_cudnn_fmha_gen_module().decode
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
        out,
        batch_offsets,
    )

    return out
