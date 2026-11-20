# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 by FlashInfer team.

# Compatibility alias: flashinfer.token_sparse moved to flashinfer.sdsa.token_sparse.
from .sdsa.token_sparse import (
    compact_nvfp4_bitmap_numel,
    compact_nvfp4_kv_cache,
    compact_nvfp4_scratch_numel,
    gather_nvfp4_kv_cache_by_position,
    group_union_masks,
    token_sparse_kv_indices,
    union_key_limit,
)

__all__ = [
    "compact_nvfp4_bitmap_numel",
    "compact_nvfp4_kv_cache",
    "compact_nvfp4_scratch_numel",
    "gather_nvfp4_kv_cache_by_position",
    "group_union_masks",
    "token_sparse_kv_indices",
    "union_key_limit",
]
