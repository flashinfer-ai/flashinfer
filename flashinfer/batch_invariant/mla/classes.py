# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The registry entries of the sparse MLA classes (``flashinfer.batch_invariant.fingerprint``)."""

from __future__ import annotations

from typing import Any, Dict

#: The APIs that run the classes under ``batch_invariant=True``.
OPERATORS = (
    "BatchSparseMLADecodePagedTSWrapper",
    "batch_sparse_mla_decode_with_paged_kv_cache",
    "trtllm_batch_decode_with_kv_cache_mla",
    "trtllm_prefill_with_kv_cache_mla",
)


def _sparse_mla_fingerprint(dtype, head_dim: int) -> str:
    from .spec import dtype_name, spec_fingerprint

    return spec_fingerprint(dtype_name(dtype), int(head_dim))


def register(classes: Dict[str, Any], class_info: Any) -> None:
    """Add the sparse MLA classes to the mode's registry: the Q / KV dtype and the query width are the dimensions; the
    head count, the list width and the batch are not."""
    for op in OPERATORS:
        classes[op] = class_info(
            "mla.sparse", ("dtype", "head_dim"), _sparse_mla_fingerprint
        )
