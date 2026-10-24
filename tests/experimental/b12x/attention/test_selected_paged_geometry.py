"""Selected-position attention with shared and per-KV-head selection lists."""

import pytest
import torch

from b12x.attention.paged._selected_forward import (
    launch_selected_paged_gqa_direct,
    launch_sparse_gqa_merge,
    launch_sparse_gqa_split,
)


@pytest.mark.parametrize("head_dim", [192, 256])
@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("selection_groups", [1, 2])
@pytest.mark.parametrize("rows", [3, 17])
@pytest.mark.parametrize("fp8", [False, True])
def test_selected_geometry(head_dim, direct, selection_groups, rows, fp8):
    torch.manual_seed(843)
    heads, kv_heads, tokens, width, splits = 8, 2, 640, 4104, 8
    q = torch.randn(rows, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(6, 128, kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    k_scale = torch.tensor([0.5], device="cuda") if fp8 else None
    v_scale = torch.tensor([2.0], device="cuda") if fp8 else None
    if fp8:
        k, v = k.to(torch.float8_e4m3fn), v.to(torch.float8_e4m3fn)
    table = (
        torch.arange(5, dtype=torch.int32, device="cuda").expand(rows, -1).contiguous()
    )
    requests = torch.arange(rows, dtype=torch.int32, device="cuda")
    positions = tokens - 1 - torch.arange(rows, dtype=torch.int64, device="cuda") * 23
    requests[-1] = positions[-1] = -1
    selected = torch.full(
        (rows, selection_groups, width), -1, dtype=torch.int32, device="cuda"
    )
    for group in range(selection_groups):
        selected[:, group, : tokens // 2] = torch.arange(
            group, tokens, 2, device="cuda"
        )
    if selection_groups == 1:
        selected = selected[:, 0].contiguous()
    output = torch.empty_like(q)
    partial = torch.empty(rows, splits, heads, head_dim, device="cuda")
    lse = torch.empty(rows, splits, heads, device="cuda")
    kwargs = dict(
        query=q,
        key_cache=k,
        value_cache=v,
        k_descale=k_scale,
        v_descale=v_scale,
        block_table=table,
        request_ids=requests,
        selected_positions=selected,
        query_positions=positions,
        softmax_scale=head_dim**-0.5,
    )

    def run():
        if direct:
            launch_selected_paged_gqa_direct(
                **kwargs, output=output, output_lse=None, kv_warps=2
            )
        else:
            launch_sparse_gqa_split(
                **kwargs, partial_output=partial, partial_lse=lse, splits=splits
            )
            launch_sparse_gqa_merge(
                partial_output=partial,
                partial_lse=lse,
                output=output,
                output_lse=None,
                rows=rows,
                splits=splits,
            )

    run()
    reference = torch.zeros_like(output)
    for row in range(rows - 1):
        for head in range(heads):
            group = head // (heads // kv_heads)
            indices = selected[row] if selection_groups == 1 else selected[row, group]
            indices = indices[(indices >= 0) & (indices <= positions[row])].long()
            keys, values = (x.float().flatten(0, 1)[indices, group] for x in (k, v))
            if fp8:
                keys, values = keys * k_scale, values * v_scale
            probability = torch.softmax(
                (keys @ q[row, head].float()) * head_dim**-0.5, -1
            )
            reference[row, head] = probability @ values
    torch.testing.assert_close(output, reference, rtol=0.02, atol=0.004)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    allocated = torch.cuda.memory_allocated()
    output.fill_(float("nan"))
    partial.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == allocated
    torch.testing.assert_close(output, reference, rtol=0.02, atol=0.004)
    assert not output[-1].any()
