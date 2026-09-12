# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.

import pytest
import torch
from flashinfer.deepseek_v41 import deepseek_v41_paged_indices


@pytest.mark.parametrize("page_size", [32, 64, 128])
@pytest.mark.parametrize("query_tokens", [1, 5])
def test_physical_pages_boundaries_and_changed_graph(page_size, query_tokens):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("requires SM100/SM103")
    torch.manual_seed(41)
    ids = torch.randint(
        -10,
        5 * page_size + 10,
        (3, query_tokens, 513),
        device="cuda",
        dtype=torch.int32,
    )
    ids[..., :8] = torch.tensor(
        [
            -1,
            0,
            page_size - 1,
            page_size,
            5 * page_size - 1,
            5 * page_size,
            2**31 - 1,
            -(2**31),
        ],
        device="cuda",
        dtype=torch.int32,
    )
    table = torch.randperm(15, device="cuda").int().reshape(3, 5)
    table[1, 2] = -1
    out = torch.empty_like(ids)

    def expected():
        page = ids.long() // page_size
        physical_page = (
            table[:, None, :].expand(3, query_tokens, 5).gather(2, page.clamp(0, 4))
        )
        valid = (ids >= 0) & (ids < 5 * page_size) & (physical_page >= 0)
        return torch.where(valid, physical_page * page_size + ids % page_size, -1)

    assert deepseek_v41_paged_indices(ids, table, page_size=page_size, out=out) is out
    torch.testing.assert_close(out, expected(), atol=0, rtol=0)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        deepseek_v41_paged_indices(ids, table, page_size=page_size, out=out)
    ids[..., 0] = page_size + 1
    table.copy_(table.flip(1))
    graph.replay()
    torch.testing.assert_close(out, expected(), atol=0, rtol=0)
    with pytest.raises(ValueError, match="overlap"):
        deepseek_v41_paged_indices(ids, table, page_size=page_size, out=ids)
    torch.cuda.set_sync_debug_mode("error")
    try:
        deepseek_v41_paged_indices(ids, table, page_size=page_size, out=out)
    finally:
        torch.cuda.set_sync_debug_mode("default")
