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
    max_token_per_sequence: int,
    actual_seq_lens_q: torch.Tensor,
    actual_seq_lens_kv: torch.Tensor,
    block_tables: torch.Tensor,
    causal: bool,
    return_lse: bool,
    batch_offsets: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    use_cuda_graph: bool = False,
) -> torch.Tensor:
    """Performs batched prefill attention with paged KV cache using cuDNN.

    Args:
        q: Query tensor of shape (Total number of tokens, num_heads_qo, head_dim)
        k_cache: Key cache tensor of shape   (total_num_pages, num_heads_kv, page_size, head_dim)
        v_cache: Value cache tensor of shape (total_num_pages, num_heads_kv, page_size, head_dim)
        scale: Scaling factor for attention scores, typically 1/sqrt(head_dim)
        workspace_buffer: Workspace buffer for cuDNN operations. Scales with batch size. 128 MB should be sufficient for most cases
        actual_seq_lens_q:  Actual number of tokens per query sequence shape (batch_size,) on cpu or device,
        actual_seq_lens_kv: Actual sequence lengths for key/values per batch, shape (batch_size,) on CPU
        block_tables: Page table mapping for KV cache, shape (batch_size, num_pages_per_seq) on GPU
        num_pages_per_seq: Number of pages allocated per sequence (max_s_kv / page_size)
        causal: Whether to apply causal masking (must be True)
        return_lse: Whether to return log-sum-exp values
        out: Optional pre-allocated output tensor
        lse: Optional pre-allocated tensor for log-sum-exp values if return_lse is True else returns None
        use_cuda_graph: Whether to use CUDA graph for the prefill operation

    Returns:
        Output tensor of shape (batch_size, seq_len_q, num_heads_qo, head_dim)
        If return_lse is True, also returns log-sum-exp tensor of shape (batch_size, seq_len_q, num_heads_qo)

    Note:
        Currently only supports causal attention (causal must be True)
        Query and KV heads can have different sizes (num_heads_qo >= num_heads_kv)
        When using cuda graph, actual_seq_lens_q and actual_seq_lens_kv must be on the same device as q
    """

    h_qo = q.shape[1]
    num_sequences = actual_seq_lens_q.shape[0]

    assert causal, "Currently only supports causal attention"

    if return_lse:
        if lse is None:
            lse = torch.empty(
                num_sequences,
                max_token_per_sequence,
                h_qo,
                device=q.device,
                dtype=torch.float32,
            )
        if lse.shape != (num_sequences, max_token_per_sequence, h_qo):
            raise ValueError(
                "lse must have shape (num_sequences, max_token_per_sequence, h_qo)"
            )

    if out is None:
        out = torch.empty_like(q)

    if actual_seq_lens_q.is_cuda == False:
        actual_seq_lens_q_gpu = actual_seq_lens_q.to(q.device)
    if actual_seq_lens_kv.is_cuda == False:
        actual_seq_lens_kv_gpu = actual_seq_lens_kv.to(q.device)

    run_func = get_cudnn_fmha_gen_module().prefill
    run_func(
        num_sequences,
        max_token_per_sequence,
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
        causal,
        return_lse,
        out,
        lse,
        batch_offsets,
        use_cuda_graph,
    )

    return out, lse
