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
    *,
    max_token_per_sequence: int,
    max_sequence_kv: int,
    actual_seq_lens_q: torch.Tensor,
    actual_seq_lens_kv: torch.Tensor,
    block_tables: Optional[torch.Tensor] = None,
    causal: bool,
    return_lse: bool,
    batch_offsets_q: Optional[torch.Tensor] = None,
    batch_offsets_o: Optional[torch.Tensor] = None,
    batch_offsets_k: Optional[torch.Tensor] = None,
    batch_offsets_v: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    is_cuda_graph_compatible: bool = False,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Performs batched prefill attention with paged KV cache using cuDNN.

    Args:
        q: Query tensor of shape (Total number of tokens, num_heads_qo, head_dim)
        k_cache: Key cache tensor of shape   (total_num_pages, num_heads_kv, page_size, head_dim) if paged kv cache is enabled else (Total sequence length of kv, num_heads_kv, d_qk)
        v_cache: Value cache tensor of shape (total_num_pages, num_heads_kv, page_size, head_dim) if paged kv cache is enabled else (Total sequence length of kv, num_heads_kv, d_vo)
        scale: Scaling factor for attention scores, typically 1/sqrt(head_dim)
        workspace_buffer: Workspace buffer for cuDNN operations. Scales with batch size. 128 MB should be sufficient for most cases
        max_token_per_sequence: Maximum number of tokens per query sequence (s_qo_max)
        max_sequence_kv: Maximum number of tokens per key/value sequence (s_kv_max)
        actual_seq_lens_q:  Actual number of tokens per query sequence shape (batch_size,) on cpu or device (cpu if cuda_graph is False)
        actual_seq_lens_kv: Actual sequence lengths for key/values per batch, shape (batch_size,) on CPU or device (cpu if cuda_graph is False)
        block_tables: Page table mapping for KV cache, shape (batch_size, num_pages_per_seq) on GPU
        causal: Whether to apply causal masking
        return_lse: Whether to return log-sum-exp values (must be True)
        out: Optional pre-allocated output tensor
        lse: Optional pre-allocated tensor for log-sum-exp values if return_lse is True else returns None
        is_cuda_graph_compatible: Whether the prefill operation is compatible with CUDA graph
        batch_offsets_q: Optional batch offsets for query tensor of shape (batch_size,) on GPU
        batch_offsets_o: Optional batch offsets for output tensor of shape (batch_size,) on GPU
        batch_offsets_k: Optional batch offsets for key tensor of shape (batch_size,) on GPU
        batch_offsets_v: Optional batch offsets for value tensor of shape (batch_size,) on GPU

    Returns:
        Output tensor of shape (batch_size * seq_len_q, num_heads_qo, head_dim)
        If return_lse is True, also returns log-sum-exp tensor of shape (batch_size, seq_len_q, num_heads_qo)

    Note:
        Query and KV heads can have different sizes (num_heads_qo >= num_heads_kv)
        When using cuda graph, actual_seq_lens_q and actual_seq_lens_kv must be on the same device as q
        Head dimension of query and key must be 128 or 192
        Head dimension of value and output must be 128
    """

    d_qk = q.shape[2]
    h_qo = q.shape[1]
    num_sequences = actual_seq_lens_q.shape[0]

    if v_cache.dim() == 3:
        d_vo = v_cache.shape[2]
    else:
        d_vo = v_cache.shape[3]

    assert return_lse, "Currently only supports return_lse = True"

    assert (d_qk == 192 and block_tables is None) or (
        d_qk == 128 and block_tables is not None
    ), "Currently only supports if d_qk = 192 and block_tables is None or d_qk = 128 and block_tables is not None"

    if max_sequence_kv is None:
        max_sequence_kv = max_token_per_sequence

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
        out_shape = (q.shape[0], h_qo, d_vo)
        out = torch.empty(out_shape, device=q.device, dtype=q.dtype)

    actual_seq_lens_q_gpu = actual_seq_lens_q.to(q.device, non_blocking=True)

    actual_seq_lens_kv_gpu = actual_seq_lens_kv.to(q.device, non_blocking=True)

    run_func = get_cudnn_fmha_gen_module().prefill
    run_func(
        num_sequences,
        max_token_per_sequence,  # max_s_qo
        max_sequence_kv,  # max_s_kv
        q,
        k_cache,
        v_cache,
        scale,
        workspace_buffer,
        actual_seq_lens_q,  # actual_seq_lens_q
        actual_seq_lens_kv,  # actual_seq_lens_kv
        actual_seq_lens_q_gpu,
        actual_seq_lens_kv_gpu,
        block_tables,
        causal,
        return_lse,
        out,
        lse,
        batch_offsets_q,
        batch_offsets_o,
        batch_offsets_k,
        batch_offsets_v,
        is_cuda_graph_compatible,
    )

    return out, lse
