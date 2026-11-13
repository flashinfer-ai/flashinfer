# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 by FlashInfer team.

# Compatibility alias: flashinfer.token_sparse moved to flashinfer.sdsa.token_sparse.
from .sdsa.token_sparse import (
    gather_nvfp4_kv_cache,
    group_union_masks,
    token_sparse_kv_indices,
    union_key_limit,
)

__all__ = [
    "gather_nvfp4_kv_cache",
    "group_union_masks",
    "token_sparse_kv_indices",
    "union_key_limit",
]
