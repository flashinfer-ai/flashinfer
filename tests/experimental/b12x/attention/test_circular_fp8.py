"""Circular attention joins FP8 history with uncommitted BF16 candidates."""

import pytest
import torch

from b12x.attention.paged._selected_forward import (
    BFloat16,
    Float32,
    Float8E4M3FN,
    Int32,
    Int64,
    _compile,
    _fake_pointer,
    _pointer,
    current_cuda_stream,
    launch_sparse_gqa_merge,
    run_compiled,
)
from b12x.testing.reference.helpers import require_b12x


@pytest.mark.parametrize("dimension", [192, 256])
@pytest.mark.parametrize("splits", [1, 8])
def test_circular_fp8_history_and_bf16_candidates(dimension, splits):
    device = require_b12x()
    torch.manual_seed(721)
    batch, width, window, heads, groups = 4, 7, 512, 8, 2
    rows = batch * width
    query = torch.randn(rows, heads, dimension, device=device).bfloat16()
    key = torch.randn(rows, groups, dimension, device=device).bfloat16()
    value = torch.randn_like(key)
    history_key = torch.randn(batch + 1, window, groups, dimension, device=device).to(
        torch.float8_e4m3fn
    )
    history_value = torch.randn_like(history_key.float()).to(torch.float8_e4m3fn)
    starts = torch.tensor([0, 510, 1027, 1024], device=device)
    requests = torch.arange(batch, device=device).repeat_interleave(width)
    positions = starts[requests] + torch.arange(width, device=device).repeat(batch)
    positions[:3] = positions[-width:] = -1
    begins = torch.arange(batch + 1, device=device, dtype=torch.int32) * width
    states = torch.arange(1, batch + 1, device=device)
    scale = torch.ones(1, device=device)
    output = torch.empty_like(query)
    partial = torch.empty(rows, splits, heads, dimension, device=device)
    lse = torch.empty(rows, splits, heads, device=device)
    _, program = _compile(
        query=query,
        key_cache=history_key,
        value_cache=history_value,
        request_ids=requests,
        direct_output=splits == 1,
        return_lse=splits != 1,
        kv_warps=2,
        selection_width=window,
        first_position=3,
    )
    continuation = tuple(
        _pointer(t, dtype)
        for t, dtype in zip(
            (key, value, starts, begins, states),
            (BFloat16, BFloat16, Int64, Int32, Int64),
            strict=True,
        )
    )

    def run():
        run_compiled(
            program,
            (
                _pointer(query, BFloat16),
                _pointer(history_key, Float8E4M3FN),
                _pointer(history_value, Float8E4M3FN),
                _pointer(scale, Float32),
                _pointer(scale, Float32),
                _fake_pointer(Int32),
                _pointer(requests, Int64),
                _fake_pointer(Int32),
                _pointer(positions, Int64),
                _pointer(partial, Float32),
                _pointer(lse, Float32),
                _pointer(output, BFloat16),
                _fake_pointer(BFloat16),
                continuation,
                batch + 1,
                batch,
                0,
                dimension**-0.5,
                rows,
                splits,
                current_cuda_stream(),
            ),
        )
        if splits > 1:
            launch_sparse_gqa_merge(
                partial_output=partial,
                partial_lse=lse,
                output=output,
                output_lse=None,
                rows=rows,
                splits=splits,
            )

    expected = torch.zeros_like(output)
    for row in range(rows - width):
        position = int(positions[row])
        if position < 0:
            continue
        request = row // width
        selected = torch.arange(
            max(3, position - window + 1), position + 1, device=device
        )
        history = selected < starts[request]
        operands = []
        for cached, candidates in ((history_key, key), (history_value, value)):
            joined = torch.cat(
                (
                    cached.float()[request + 1, selected[history] % window],
                    candidates.float()[
                        request * width + selected[~history] - starts[request]
                    ],
                )
            )
            operands.append(
                joined.repeat_interleave(heads // groups, dim=1).transpose(0, 1)
            )
        logits = (
            torch.einsum("hd,hsd->hs", query[row].float(), operands[0])
            * dimension**-0.5
        )
        expected[row] = torch.einsum(
            "hs,hsd->hd", logits.softmax(-1), operands[1]
        ).bfloat16()
    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    output.fill_(float("nan"))
    partial.fill_(float("nan"))
    allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
    graph.replay()
    torch.cuda.synchronize()
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
    torch.testing.assert_close(output.float(), expected.float(), atol=0.004, rtol=0.02)
    graph.reset()
