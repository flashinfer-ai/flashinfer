# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate direct router offsets against independent CPU group membership."""

import pytest
import torch

from flashinfer.fused_moe.cute_dsl.moe_utils import (
    _get_moe_utils_module,
    allocate_moe_sort_buffers,
    moe_sort,
)


@pytest.mark.parametrize(
    "tokens,experts,top_k,tile,local_start,local_count",
    [
        (1, 8, 2, 1, 0, 8),
        (17, 8, 2, 1, 0, 8),
        (64, 8, 2, 128, 0, 8),
        (1024, 32, 4, 1, 0, 32),
        (1025, 32, 4, 1, 0, 32),
        (4096, 32, 4, 1, 0, 32),
        (4096, 32, 4, 16, 4, 8),
        (512, 128, 8, 1, 0, 128),
        (17, 384, 8, 1, 0, 384),
        (4096, 512, 8, 1, 0, 512),
        (65536, 32, 4, 1, 0, 32),
    ],
)
def test_direct_offsets_live_capture(
    tokens, experts, top_k, tile, local_start, local_count
):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100-family GPU required")
    # Unique experts per token, skew and empty leading/interior/trailing groups.
    cpu_ids = (
        (torch.arange(tokens).unsqueeze(1) % 3) * top_k + torch.arange(top_k)
    ) % experts
    ids = cpu_ids.to(device="cuda", dtype=torch.int32)
    weights = torch.ones_like(ids, dtype=torch.float32) / top_k
    original_weights = weights.clone()
    buffers = allocate_moe_sort_buffers(tokens, experts, top_k, local_count, tile)
    counts = torch.empty(2 * experts, device="cuda", dtype=torch.int32)
    guarded_offsets = torch.full((experts + 3,), 7919, device="cuda", dtype=torch.int32)
    offsets = guarded_offsets[1:-1]
    module = _get_moe_utils_module()
    native = module["flashinfer_moe_sort_with_offsets"]

    def forward():
        native(
            ids.data_ptr(),
            weights.data_ptr(),
            tokens,
            experts,
            top_k,
            local_start,
            local_count,
            tile,
            False,
            buffers["out_tile_idx_to_expert_idx"].data_ptr(),
            buffers["out_tile_idx_to_mn_limit"].data_ptr(),
            buffers["out_expanded_idx_to_permuted_idx"].data_ptr(),
            buffers["out_permuted_idx_to_expanded_idx"].data_ptr(),
            buffers["out_total_num_padded_tokens"].data_ptr(),
            buffers["out_num_non_exiting_tiles"].data_ptr(),
            counts.data_ptr() if tokens > 1024 else 0,
            torch.cuda.current_stream().cuda_stream,
            offsets.data_ptr(),
        )

    def validate(expected_ids):
        flat = expected_ids.flatten()
        histogram = torch.bincount(flat, minlength=experts).to(torch.int32)
        histogram[:local_start] = 0
        histogram[local_start + local_count :] = 0
        padded = ((histogram + tile - 1) // tile) * tile
        expected_offsets = torch.cat(
            (torch.zeros(1, dtype=torch.int32), padded.cumsum(0).to(torch.int32))
        )
        torch.testing.assert_close(offsets.cpu(), expected_offsets, rtol=0, atol=0)
        assert guarded_offsets[0].item() == guarded_offsets[-1].item() == 7919
        assert (
            buffers["out_total_num_padded_tokens"].item() == expected_offsets[-1].item()
        )
        inverse = buffers["out_expanded_idx_to_permuted_idx"].flatten().cpu().long()
        permutation = buffers["out_permuted_idx_to_expanded_idx"].cpu().long()
        for expert in range(experts):
            begin = expected_offsets[expert].item()
            count = histogram[expert].item()
            if not count:
                continue
            members = permutation[begin : begin + count]
            expected_members = torch.nonzero(flat == expert).flatten()
            torch.testing.assert_close(
                members.sort().values, expected_members, rtol=0, atol=0
            )
            torch.testing.assert_close(
                inverse[members], torch.arange(begin, begin + count), rtol=0, atol=0
            )
        torch.testing.assert_close(weights, original_weights, rtol=0, atol=0)
        torch.testing.assert_close(ids.cpu().long(), expected_ids, rtol=0, atol=0)

    forward()
    validate(cpu_ids)
    initial_offsets = offsets.clone()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        forward()
    torch.cuda.current_stream().wait_stream(side)
    for shift in (1, experts - 1, 2):
        changed_ids = (cpu_ids + shift) % experts
        ids.copy_(changed_ids)
        for tensor in buffers.values():
            tensor.fill_(-12345)
        counts.fill_(-12345)
        offsets.fill_(-12345)
        graph.replay()
        validate(changed_ids)
    if tokens == 17 and experts == 8:
        assert not torch.equal(initial_offsets, offsets), (
            "Changed routing negative control"
        )


def test_direct_offsets_wrapper_validation():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100-family GPU required")
    ids = torch.tensor([[0, 1]], device="cuda", dtype=torch.int32)
    weights = torch.ones((1, 2), device="cuda")
    for offsets in (
        torch.empty(8, device="cuda", dtype=torch.int32),
        torch.empty(9, device="cuda", dtype=torch.int64),
        torch.empty(18, device="cuda", dtype=torch.int32)[::2],
        torch.empty(9, device="cpu", dtype=torch.int32),
    ):
        with pytest.raises(ValueError, match="out_expert_first_token_offset"):
            moe_sort(
                ids,
                weights,
                8,
                2,
                tile_tokens_dim=1,
                out_expert_first_token_offset=offsets,
            )
    offsets = torch.empty(9, device="cuda", dtype=torch.int32)
    result = moe_sort(
        ids, weights, 8, 2, tile_tokens_dim=1, out_expert_first_token_offset=offsets
    )
    assert len(result) == 6
    torch.testing.assert_close(
        offsets.cpu(),
        torch.tensor([0, 1, 2, 2, 2, 2, 2, 2, 2], dtype=torch.int32),
        rtol=0,
        atol=0,
    )
    # Existing consumers can keep the original 17-argument native ABI.
    expected = [tensor.clone() for tensor in result]
    for tensor in result:
        tensor.fill_(-1)
    offsets.fill_(7919)
    _get_moe_utils_module()["flashinfer_moe_sort"](
        ids.data_ptr(),
        weights.data_ptr(),
        1,
        8,
        2,
        0,
        8,
        1,
        False,
        *(tensor.data_ptr() for tensor in result),
        0,
        torch.cuda.current_stream().cuda_stream,
    )
    for actual, reference in zip(result, expected, strict=True):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    assert torch.all(offsets == 7919)
