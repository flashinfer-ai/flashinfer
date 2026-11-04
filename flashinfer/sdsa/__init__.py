# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 by FlashInfer team.

"""SDSA (Simplified DSA): sparse attention in which a one-head indexer selects
the Top-K keys of each query token and an MQA core attends to these keys."""

from .indexer import (
    gather_key_sample,
    indexer_scores,
    indexer_threshold_select,
    quantize_write_index_qk_nvfp4,
    thresholds_from_sample,
    top_k_from_candidates,
)

__all__ = [
    "gather_key_sample",
    "indexer_scores",
    "indexer_threshold_select",
    "quantize_write_index_qk_nvfp4",
    "thresholds_from_sample",
    "top_k_from_candidates",
]
