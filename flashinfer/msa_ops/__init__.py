"""Minimax Sparse Attention (MSA) operations. SM120/SM121 (Blackwell) only."""

from .proxy_score import (
    msa_proxy_score,
    msa_proxy_score_fp4,
)
from .sparse_prefill import msa_sparse_attention
from .sparse_decode import msa_sparse_decode_attention
from .sparse_topk_select import msa_topk_select

# Capability flag for callers (e.g. vLLM): the sparse attention ops accept K/V
# views split from a paged cache that packs K and V in one 2*head_dim content
# dim per token.
SUPPORTS_PACKED_KV = True

__all__ = [
    "SUPPORTS_PACKED_KV",
    "msa_proxy_score",
    "msa_proxy_score_fp4",
    "msa_sparse_attention",
    "msa_sparse_decode_attention",
    "msa_topk_select",
]
