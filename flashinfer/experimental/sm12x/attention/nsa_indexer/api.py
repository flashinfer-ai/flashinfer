# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
"""Public surface for attention.nsa_indexer (docs in the op ``__init__``).

Naming: uniform ``_paged`` / ``_contiguous`` suffixes replace the upstream
``msa_`` / ``paged_`` / ``contiguous_`` prefix mix; the pipeline stage is the
verb (quantize -> logits/block_scores -> topk/q2k_indices).
"""

from __future__ import annotations

from ..._lib.gating import default_is_supported
from ._impl import (
    IndexerContiguousMetadata as ContiguousMetadata,
)
from ._impl import (
    IndexerPagedDecodeMetadata as PagedDecodeMetadata,
)
from ._impl import (
    build_paged_mqa_schedule_metadata as plan_paged_schedule,
)
from ._impl import (
    clear_indexer_caches as clear_caches,
)
from ._impl import (
    contiguous_logits as logits_contiguous,
)
from ._impl import (
    contiguous_tiled_topk as topk_tiled,
)
from ._impl import (
    msa_contiguous_block_scores as block_scores_contiguous,
)
from ._impl import (
    msa_decode_query_positions as query_positions_decode,
)
from ._impl import (
    msa_paged_decode_block_scores as block_scores_paged,
)
from ._impl import (
    msa_prefill_query_positions as query_positions_prefill,
)
from ._impl import (
    msa_q2k_indices_decode as q2k_indices_decode,
)
from ._impl import (
    msa_q2k_indices_prefill as q2k_indices_prefill,
)
from ._impl import (
    msa_topk_blocks as topk_blocks,
)
from ._impl import (
    paged_decode_logits as logits_paged,
)
from ._impl import (
    quantize_msa_q_fp8 as quantize_q_fp8,
)
from ._impl import (
    resolve_contiguous_prefill_block_k,
)
from ._impl import (
    uses_paged_mqa_schedule as uses_paged_schedule,
)
from .kernel import (
    IndexerOutputMode as OutputMode,
)
from .kernel import (
    IndexerScoreMode as ScoreMode,
)
from .paged import (
    INDEX_HEAD_DIM,
    PAGED_INDEX_PAGE_SIZE,
    index_topk_fp8,
    resolve_local_num_q_heads,
    resolve_replicated_num_q_heads,
)
from .paged import (
    IndexerPagedMetadata as PagedMetadata,
)
from .paged import (
    prepare_paged_indexer_metadata as prepare_paged_metadata,
)
from .persistent_topk import (
    SM12XPersistentTopK2048Binding as PersistentTopK2048Binding,
)
from .persistent_topk import (
    SM12XPersistentTopK2048ScratchCaps as PersistentTopK2048Caps,
)
from .persistent_topk import (
    SM12XPersistentTopK2048ScratchPlan as PersistentTopK2048Plan,
)
from .persistent_topk import (
    build_persistent_topk2048_binding as bind_persistent_topk2048,
)
from .persistent_topk import (
    persistent_topk2048_scratch_nbytes,
)
from .persistent_topk import (
    plan_persistent_topk2048_scratch as plan_persistent_topk2048,
)
from .persistent_topk import (
    run_persistent_topk2048,
    supports_persistent_topk2048,
)
from .scratch import (
    INDEXER_SOURCE_LAYOUT_CONTIGUOUS as SOURCE_LAYOUT_CONTIGUOUS,
)
from .scratch import (
    INDEXER_SOURCE_LAYOUT_PAGED as SOURCE_LAYOUT_PAGED,
)
from .scratch import (
    SM12XIndexerContiguousBinding as ContiguousBinding,
)
from .scratch import (
    SM12XIndexerContiguousScratch as ContiguousScratch,
)
from .scratch import (
    SM12XIndexerMSAContiguousBinding as MSAContiguousBinding,
)
from .scratch import (
    SM12XIndexerMSAPagedBinding as MSAPagedBinding,
)
from .scratch import (
    SM12XIndexerPagedBinding as PagedBinding,
)
from .scratch import (
    SM12XIndexerPagedScratch as PagedScratch,
)
from .scratch import (
    SM12XIndexerScratchCaps as Caps,
)
from .scratch import (
    SM12XIndexerScratchPlan as Plan,
)
from .scratch import (
    build_indexer_contiguous_binding as bind_contiguous,
)
from .scratch import (
    build_indexer_msa_contiguous_binding as bind_msa_contiguous,
)
from .scratch import (
    build_indexer_msa_paged_binding as bind_msa_paged,
)
from .scratch import (
    build_indexer_paged_binding as bind_paged,
)
from .scratch import (
    plan_indexer_scratch as plan,
)
from . import META


def is_supported(device=None) -> bool:
    """True on SM120/SM121 with nvidia-cutlass-dsl >= 4.6.0 and triton."""
    return default_is_supported(device, requires=META.requires)


__all__ = list(META.entry_points)
