"""Qwen4Exp quantized sparse attention (QSA) operations.

The pre-indexer, scorer, route expansion, selection, paged attention runtime
and output gate of a QSA layer. The scorer needs SM80 or newer.
"""

from .attention import QSAAttention
from .capabilities import (
    QSA_CAP_ATTENTION_PAGED,
    QSA_CAP_FP8,
    QSA_CAP_NVFP4,
    QSA_CAP_OUTPUT_GATE,
    QSA_CAP_SELECTION,
    qsa_capabilities,
    qsa_capability_names,
)
from .output_gate import qsa_output_gate
from .pre_indexer import (
    QSA_PRE_INDEXER_NARROW_E4M3,
    QSA_PRE_INDEXER_SAME_AS_COMPUTE,
    qsa_pre_indexer,
    qsa_pre_indexer_dispatch_mask,
)
from .route import (
    qsa_expand_block_route,
    qsa_route_from_blocks,
    qsa_route_from_logical,
)
from .runtime import QSA, QSAConfig, QSAWorkspaceRequirements
from .scores import qsa_paged_scores
from .selection import QSASelection

__all__ = [
    "QSA",
    "QSAAttention",
    "QSAConfig",
    "QSASelection",
    "QSAWorkspaceRequirements",
    "QSA_CAP_ATTENTION_PAGED",
    "QSA_CAP_FP8",
    "QSA_CAP_NVFP4",
    "QSA_CAP_OUTPUT_GATE",
    "QSA_CAP_SELECTION",
    "QSA_PRE_INDEXER_NARROW_E4M3",
    "QSA_PRE_INDEXER_SAME_AS_COMPUTE",
    "qsa_capabilities",
    "qsa_capability_names",
    "qsa_expand_block_route",
    "qsa_output_gate",
    "qsa_paged_scores",
    "qsa_pre_indexer",
    "qsa_pre_indexer_dispatch_mask",
    "qsa_route_from_blocks",
    "qsa_route_from_logical",
]
