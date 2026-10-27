"""Attention output gating preserves the separate BF16 rounding boundaries."""

import pytest
import torch

from b12x.attention.paged._selected_forward import (
    _output_gate_stride,
    launch_selected_paged_gqa_direct,
    launch_sparse_gqa_merge,
    launch_sparse_gqa_split,
)
from b12x.testing.reference.helpers import require_b12x


@pytest.mark.parametrize(
    "shape,stride,dtype",
    [
        ((3, 8), (8, 1), torch.bfloat16),
        ((4, 7), (7, 1), torch.bfloat16),
        ((4, 8), (16, 2), torch.bfloat16),
        ((4, 8), (8, 1), torch.float32),
    ],
)
def test_rejects_incompatible_gate(shape, stride, dtype):
    output = torch.empty(4, 8, 192, device="meta", dtype=torch.bfloat16)
    gate = torch.empty_strided(shape, stride, device="meta", dtype=dtype)
    with pytest.raises(ValueError, match="output_gate must be BF16"):
        _output_gate_stride(gate, output)


@pytest.mark.parametrize("head_dim", [192, 256])
@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("rows", [4, 65])
@pytest.mark.parametrize("fp8", [False, True])
def test_output_gate_rounding_and_replay(head_dim, direct, rows, fp8):
    device = require_b12x()
    torch.manual_seed(1283)
    heads, groups, splits = 8, 2, 8
    q = torch.randn(rows, heads, head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(2, 128, groups, head_dim, device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    descale = torch.ones(1, device=device) if fp8 else None
    if fp8:
        k, v = k.to(torch.float8_e4m3fn), v.to(torch.float8_e4m3fn)
    pages = torch.arange(2, device=device, dtype=torch.int32)[None]
    requests = torch.zeros(rows, device=device, dtype=torch.int64)
    positions = torch.full((rows,), 192, device=device, dtype=torch.int64)
    requests[-1] = positions[-1] = -1
    selected = torch.full((rows, groups, 4104), -1, device=device, dtype=torch.int32)
    selected[..., :256] = torch.arange(256, device=device)
    packed = torch.randn(rows, heads * 3, device=device, dtype=torch.bfloat16)
    gate = packed[:, heads : 2 * heads]
    gate[0] = torch.tensor([-100, -10, -1, 0, 1, 10, 100, 0.5], device=device)
    output, ungated = torch.empty_like(q), torch.empty_like(q)
    partial = torch.empty(rows, splits, heads, head_dim, device=device)
    lse = torch.empty(rows, splits, heads, device=device)
    common = dict(
        query=q,
        key_cache=k,
        value_cache=v,
        k_descale=descale,
        v_descale=descale,
        block_table=pages,
        request_ids=requests,
        query_positions=positions,
        selected_positions=selected,
        softmax_scale=head_dim**-0.5,
    )

    def run(out, logits):
        if direct:
            launch_selected_paged_gqa_direct(
                **common, output=out, output_lse=None, kv_warps=2, output_gate=logits
            )
        else:
            launch_sparse_gqa_split(
                **common, partial_output=partial, partial_lse=lse, splits=splits
            )
            launch_sparse_gqa_merge(
                partial_output=partial,
                partial_lse=lse,
                output=out,
                output_lse=None,
                rows=rows,
                splits=splits,
                output_gate=logits,
            )

    run(ungated, None)
    run(output, gate)
    torch.testing.assert_close(
        output, ungated * gate.sigmoid()[..., None], rtol=0, atol=0
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(output, gate)
    gate.neg_()
    expected = ungated * gate.sigmoid()[..., None]
    output.fill_(float("nan"))
    partial.fill_(float("nan"))
    allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert not output[-1].any()
    graph.reset()
