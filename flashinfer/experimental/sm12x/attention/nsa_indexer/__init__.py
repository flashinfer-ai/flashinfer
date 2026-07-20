# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
"""NSA indexer for SM12x: the index stage of Native Sparse Attention.

A three-stage pipeline whose outputs feed ``attention.sparse_mla`` /
``attention.compressed_mla``:

1. quantize — ``quantize_q_fp8`` (index-Q to FP8 e4m3, MXFP8 tiling;
   INDEX_HEAD_DIM 128).
2. score — MQA logits between index-Q and the FP8 index-K cache:
   ``logits_paged`` / ``logits_contiguous``, block-score reduction via
   ``block_scores_paged`` / ``block_scores_contiguous``.
3. select — ``topk_blocks`` / ``topk_tiled`` and the q2k index builders
   ``q2k_indices_decode`` / ``q2k_indices_prefill`` (+ query-position
   helpers).

Planning: one ``Caps``/``plan`` pair sizes the caller-owned scratch; facet
binds produce ``PagedBinding`` / ``ContiguousBinding`` / ``MSAPagedBinding``
/ ``MSAContiguousBinding`` (views only, capture safe). ``plan_paged_schedule``
builds the paged-MQA schedule metadata. The persistent top-k-2048 selector
ships as its own facet (``*_persistent_topk2048``). Paged index-K packing
helpers (``PagedMetadata``, ``prepare_paged_metadata``, ``index_topk_fp8``)
round out the cache-side contract.

Pure-torch semantics live in ``reference.py`` and ``msa_reference.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="nsa_indexer",
    group="attention",
    api_style="planned",
    entry_points=(
        # planning + scratch
        "Caps",
        "Plan",
        "PagedScratch",
        "ContiguousScratch",
        "PagedBinding",
        "ContiguousBinding",
        "MSAPagedBinding",
        "MSAContiguousBinding",
        "plan",
        "bind_paged",
        "bind_contiguous",
        "bind_msa_paged",
        "bind_msa_contiguous",
        # stage 1: quantize
        "quantize_q_fp8",
        # stage 2: score
        "logits_paged",
        "logits_contiguous",
        "block_scores_paged",
        "block_scores_contiguous",
        "ScoreMode",
        "OutputMode",
        # stage 3: select
        "topk_blocks",
        "topk_tiled",
        "q2k_indices_decode",
        "q2k_indices_prefill",
        "query_positions_decode",
        "query_positions_prefill",
        "resolve_contiguous_prefill_block_k",
        # paged schedule + cache-side contract
        "plan_paged_schedule",
        "uses_paged_schedule",
        "PagedMetadata",
        "PagedDecodeMetadata",
        "ContiguousMetadata",
        "prepare_paged_metadata",
        "index_topk_fp8",
        "resolve_local_num_q_heads",
        "resolve_replicated_num_q_heads",
        "INDEX_HEAD_DIM",
        "PAGED_INDEX_PAGE_SIZE",
        "SOURCE_LAYOUT_PAGED",
        "SOURCE_LAYOUT_CONTIGUOUS",
        # persistent top-k facet
        "PersistentTopK2048Caps",
        "PersistentTopK2048Plan",
        "PersistentTopK2048Binding",
        "plan_persistent_topk2048",
        "bind_persistent_topk2048",
        "run_persistent_topk2048",
        "supports_persistent_topk2048",
        "persistent_topk2048_scratch_nbytes",
        # maintenance
        "is_supported",
        "clear_caches",
    ),
    dtypes=("bf16", "fp8_e4m3"),
    recipes=("dsv4", "glm_nsa", "msa"),
    requires=("triton",),
    provenance=Provenance(
        repo="https://github.com/lukealonso/b12x",
        commit="6627d342",
        paths=("b12x/attention/indexer/",),
    ),
    test_path="tests/experimental/attention/test_nsa_indexer.py",
    since="0.7.0",
)

if TYPE_CHECKING:  # static analysis only; runtime resolution is lazy
    from .api import *  # noqa: F401,F403

install_lazy_api(globals(), META)
