"""
Minimax Sparse Attention (MSA) operations for FlashInfer.

Supports SM120 and SM121 (Blackwell) GPUs.

Phase 1: sparse_topk_select — top-K KV block selection from proxy attention scores.
Phase 2: build_k2q_csr — invert q->k top-K indices into a KV-major CSR structure.
Phase 3: sparse_attention — sparse forward attention over selected KV blocks.
"""

from .sparse_attention import sparse_attention, sparse_attention_kvmajor
from .sparse_decode import build_decode_schedule, sparse_decode_attention
from .sparse_index_utils import (
    SparseAttentionSchedule,
    build_k2q_csr,
    build_k2q_csr_schedule,
)
from .sparse_topk_select import sparse_topk_select

__all__ = [
    "SparseAttentionSchedule",
    "build_decode_schedule",
    "build_k2q_csr",
    "build_k2q_csr_schedule",
    "sparse_attention",
    "sparse_attention_kvmajor",
    "sparse_decode_attention",
    "sparse_topk_select",
]
