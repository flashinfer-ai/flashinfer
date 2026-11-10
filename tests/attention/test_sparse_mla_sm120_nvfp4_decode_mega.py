# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NVFP4 split decode: both CTA schedules, ring reuse, and fused boundaries."""

import pytest
import torch

from tests.attention.sparse_mla_prefill_utils import (
    decode_fixture,
    native_prefill_calls,
    numerical_metrics,
)
from tests.attention.test_sparse_mla_sm120_nvfp4_prefill_mega import (
    quantized_query_reference_lse,
)


@pytest.mark.parametrize("requests", [1, 2])
@pytest.mark.parametrize("chunks_per_block", [2, 10])
@pytest.mark.parametrize("offset", [0, 2])
def test_nvfp4_decode_mega_boundaries(requests, chunks_per_block, offset):
    # T7 uses 16-head CTAs; T14 uses 64-head CTAs. Ten chunks reuse both rings
    # and store directly; two chunks require the split-result merge.
    f = decode_fixture(requests, cache_format="nvfp4", history_tokens=1024)
    if offset:
        storage = torch.empty(
            f["q"].numel() + offset, device="cuda", dtype=torch.bfloat16
        )
        q = storage[offset:].view_as(f["q"])
        q.copy_(f["q"])
        f["q"] = q
    mega, baseline, lse = native_prefill_calls(
        f, streaming_prefill=False, chunks_per_block=chunks_per_block
    )
    mega()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mega()
    rows = torch.arange(f["q"].shape[0], device="cuda")
    for length in (0, 1, 31, 32, 33, 63, 65):
        for lens, ids in zip(f["lengths"], f["indices"], strict=True):
            lens[0] = 0
            lens[1] = length
            ids[2, ::7] = -1
            ids[3, :] = -1
        f["positions"].add_(1)
        graph.replay()
        metrics = numerical_metrics(f, lse, rows)
        assert metrics["finite_output"]
        assert metrics["relative_l2"] < 0.20, metrics
        expected_lse = quantized_query_reference_lse(f, rows)
        torch.testing.assert_close(lse.double(), expected_lse, atol=0.02, rtol=0)
        fused = f["out"].float().clone(), f["scales"].clone(), lse.clone()
        baseline()
        for actual, expected in zip(
            (f["out"].float(), f["scales"], lse), fused, strict=True
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
