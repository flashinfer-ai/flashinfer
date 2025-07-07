import functools
from typing import Optional

import torch

from ..jit import get_cudnn_fmha_gen_module


def cudnn_batch_decode_with_kv_cache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    scale: float,
    workspace_buffer: torch.Tensor,
    *,
    max_sequence_kv: int,
    actual_seq_lens_kv: Optional[torch.Tensor] = None,
    block_tables: Optional[torch.Tensor] = None,
    is_cuda_graph_compatible: bool = False,
    batch_offsets_q: Optional[torch.Tensor] = None,
    batch_offsets_o: Optional[torch.Tensor] = None,
    batch_offsets_k: Optional[torch.Tensor] = None,
    batch_offsets_v: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Performs batched decode attention with paged KV cache using cuDNN.

    Args:
        q: Query tensor of shape (batch_size, num_heads_qo, head_dim), seq_len_q is the maximum sequence length of queries in the batch
        k_cache: Key cache tensor of shape   (total_num_pages, num_heads_kv, page_size, head_dim)
        v_cache: Value cache tensor of shape (total_num_pages, num_heads_kv, page_size, head_dim)
        scale: Scaling factor for attention scores, typically 1/sqrt(head_dim)
        workspace_buffer: Workspace buffer for cuDNN operations. Scales with batch size. 128 MB should be sufficient for most cases
        max_sequence_kv: Maximum number of tokens per key/value sequence (s_kv_max)
        actual_seq_lens_kv: Actual sequence lengths for key/values per batch, shape (batch_size,) on CPU
        block_tables: Page table mapping for KV cache, shape (batch_size, num_pages_per_seq) on GPU
        is_cuda_graph_compatible: Whether the decode operation is compatible with CUDA graph
        batch_offsets: Optional batch offsets tensor of shape (batch_size,) on GPU
        out: Optional pre-allocated output tensor
        lse: Optional pre-allocated tensor for log-sum-exp values if return_lse is True else returns None
        batch_offsets_q: Optional batch offsets for query tensor of shape (batch_size,) on GPU
        batch_offsets_o: Optional batch offsets for output tensor of shape (batch_size,) on GPU
        batch_offsets_k: Optional batch offsets for key tensor of shape (batch_size,) on GPU
        batch_offsets_v: Optional batch offsets for value tensor of shape (batch_size,) on GPU

    Returns:
        Output tensor of shape (batch_size, num_heads_qo, head_dim)

    Note:
        Currently only supports causal attention (causal must be True)
        All tensors must be contiguous and on the same CUDA device
        Query and KV heads can have different sizes (num_heads_qo >= num_heads_kv)
    """

    bs = q.shape[0]
    s_q = 1
    h_qo = q.shape[1]
    d_vo = v_cache.shape[3]

    if out is None:
        out = torch.empty(bs, h_qo, d_vo, device=q.device, dtype=q.dtype)

    actual_seq_lens_kv_gpu = actual_seq_lens_kv.to(q.device, non_blocking=True)

    run_func = get_cudnn_fmha_gen_module().decode
    run_func(
        max_sequence_kv,
        q,
        k_cache,
        v_cache,
        scale,
        workspace_buffer,
        actual_seq_lens_kv,
        actual_seq_lens_kv_gpu,
        block_tables,
        out,
        batch_offsets_q,
        batch_offsets_o,
        is_cuda_graph_compatible,
    )

    return out
