"""
Minimax Sparse Attention (MSA) operations for FlashInfer.

Supports SM120 and SM121 (Blackwell) GPUs.
"""

from .proxy_score import (
    msa_proxy_score,
    msa_proxy_score_fp4,
    quantize_bf16_qk_to_nvfp4,
)
from .sparse_attention import msa_sparse_attention
from .sparse_decode import msa_sparse_decode_attention
from .sparse_index_utils import (
    MsaAttentionSchedule,
    msa_build_k2q_csr,
)
from .sparse_topk_select import msa_topk_select

__all__ = [
    "MsaAttentionSchedule",
    "msa_build_k2q_csr",
    "msa_proxy_score",
    "msa_proxy_score_fp4",
    "msa_sparse_attention",
    "msa_sparse_decode_attention",
    "msa_topk_select",
    "quantize_bf16_qk_to_nvfp4",
]
