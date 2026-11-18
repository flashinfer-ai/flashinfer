# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

"""Native NVFP4 mega Direct, Merge2 and General output routes."""

import pytest
import torch

from tests.attention.sparse_mla_prefill_utils import fixture, native_prefill_calls


@pytest.mark.parametrize("tokens", [7, 14])
@pytest.mark.parametrize("chunks_per_block", [2, 5, 10])
@pytest.mark.parametrize("offset", [0, 2])
def test_nvfp4_decode_mega_query_alignment(tokens, chunks_per_block, offset):
    # Ten 64-candidate chunks select General, Merge2 and Direct respectively.
    f = fixture(tokens, 64, "nvfp4")
    from flashinfer.jit.mla import gen_sparse_mla_sm120_module

    module = gen_sparse_mla_sm120_module().build_and_load()
    mega, baseline, lse = native_prefill_calls(
        f, module, streaming_prefill=False, chunks_per_block=chunks_per_block
    )
    baseline()
    expected = f["out"].float().clone(), f["scales"].clone(), lse.clone()
    storage = torch.empty(f["q"].numel() + offset, device="cuda", dtype=torch.bfloat16)
    query = storage[offset:].view_as(f["q"])
    query.copy_(f["q"])
    f["q"] = query

    mega, _, lse = native_prefill_calls(
        f, module, streaming_prefill=False, chunks_per_block=chunks_per_block
    )
    mega()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mega()
    graph.replay()
    for actual, reference in zip(
        (f["out"].float(), f["scales"], lse), expected, strict=True
    ):
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)
