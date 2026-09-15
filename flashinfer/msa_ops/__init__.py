"""Minimax Sparse Attention operations.

Sparse prefill, sparse decode, and top-k selection support SM100, SM103, SM120,
and SM121 Blackwell GPUs. Proxy scoring remains SM120/SM121-only.
"""

import torch

from ..utils import get_compute_capability
from ._blackwell_sm100 import MSASparseAttentionWorkspace
from .proxy_score import (
    msa_proxy_score,
    msa_proxy_score_fp4,
)
from .sparse_prefill import msa_sparse_attention
from .sparse_decode import msa_sparse_decode_attention
from .sparse_topk_select import msa_topk_select

# Legacy aggregate capability flag retained for callers that only target
# SM120/SM121. Mixed-architecture callers should query supports_packed_kv().
SUPPORTS_PACKED_KV = True


def supports_packed_kv(device: torch.device | str) -> bool:
    """Return whether MSA accepts packed paged K/V views on ``device``."""

    normalized_device = torch.device(device)
    return normalized_device.type == "cuda" and get_compute_capability(
        normalized_device
    ) in {(12, 0), (12, 1)}


__all__ = [
    "MSASparseAttentionWorkspace",
    "SUPPORTS_PACKED_KV",
    "msa_proxy_score",
    "msa_proxy_score_fp4",
    "msa_sparse_attention",
    "msa_sparse_decode_attention",
    "msa_topk_select",
    "supports_packed_kv",
]
