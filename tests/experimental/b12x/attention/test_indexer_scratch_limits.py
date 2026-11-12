"""Fail-closed planning limits of the indexer scratch layouts."""

from __future__ import annotations

import pytest
import torch

from b12x.attention.dsa_indexer.scratch import (
    INDEXER_SOURCE_LAYOUT_CONTIGUOUS,
    B12XIndexerScratchCaps,
    plan_indexer_scratch,
)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_contiguous_logits_tile_beyond_kernel_extent_fails_closed() -> None:
    """A tile over every K row of a large batch cannot be launched: reject it."""
    caps = dict(
        device=_device(),
        source_layout=INDEXER_SOURCE_LAYOUT_CONTIGUOUS,
        num_q_heads=32,
        max_q_rows=4_096,
        max_k_rows=32 * 16_384,
        topk=2_048,
    )
    with pytest.raises(ValueError, match="kernel tensor limit"):
        plan_indexer_scratch(B12XIndexerScratchCaps(**caps))
    bounded = plan_indexer_scratch(B12XIndexerScratchCaps(**caps, supertile_k=32_768))
    assert int(bounded.inner.layout.tile_logits_elements) == 4_096 * 32_768
