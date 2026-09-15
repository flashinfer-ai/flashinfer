"""Compound low-batch FP8 request-order plans and public graph replay."""

import dataclasses
import math
import flashinfer
import flashinfer.cake_fmha as api
import pytest
import torch
from flashinfer.jit.cake_fmha_request_ordered import (
    get_cake_fmha_request_ordered_module_spec,
)


def fp8_plan(lengths, **kwargs):
    return flashinfer.plan_cake_fmha_request_ordered_paged_decode(
        lengths,
        6,
        num_q_heads=32,
        num_kv_heads=2,
        query_dtype=torch.float8_e4m3fn,
        **kwargs,
    )


@pytest.mark.parametrize("batch,splits", [(8, 8), (27, 2), (32, 2)])
@pytest.mark.parametrize("write_lse", [False, True])
def test_fp8q_split_plan_and_binding(batch, splits, write_lse):
    plan = fp8_plan([257] * batch, write_lse=write_lse)
    assert plan == fp8_plan([257] * batch, write_lse=write_lse, num_kv_splits=splits)
    assert api._is_authenticated_request_ordered_plan(plan)
    assert plan.workspace_parts == splits and plan.reducer_grid == (
        batch * 6 * 32,
        1,
        1,
    )
    assert (
        get_cake_fmha_request_ordered_module_spec(plan.module_name).tma_workspace_bytes
        == 384
    )
    assert (
        get_cake_fmha_request_ordered_module_spec(
            plan.reducer_module_name
        ).tma_workspace_bytes
        == 0
    )
    assert not api._is_authenticated_request_ordered_plan(
        dataclasses.replace(plan, reducer_module_name="unknown")
    )
    with pytest.raises(ValueError):
        fp8_plan([257] * batch, num_kv_splits=3)
    q = torch.empty((batch * 6, 32, 256), dtype=torch.float8_e4m3fn)
    k = torch.empty((8, 64, 2, 256), dtype=torch.float8_e4m3fn).permute(0, 2, 1, 3)
    v = torch.empty_like(k)
    out = torch.empty(q.shape, dtype=torch.bfloat16)
    lse = torch.empty(q.shape[:-1]) if write_lse else None
    workspace = torch.empty(api._fp8q_split_workspace_bytes(plan), dtype=torch.uint8)
    table = torch.empty((batch, 8), dtype=torch.int32)
    lens = torch.full((batch,), 257, dtype=torch.int32)
    order = torch.arange(batch, dtype=torch.int32)
    qk = torch.tensor([0.01])
    pv = torch.tensor([1.0])
    producer, reducer = api._fp8q_split_request_ordered_arguments(
        plan, q, k, v, out, lse, workspace, order, table, lens, qk, pv
    )
    for index, tensor in (
        (0, q),
        (1, k),
        (2, v),
        (3, out),
        (5, order),
        (7, table),
        (8, lens),
        (9, qk),
        (10, pv),
    ):
        assert producer[index].data_ptr() == tensor.data_ptr()
    rows = batch * 6 * 32 * splits
    begin = workspace.data_ptr() + 384
    assert [tensor.data_ptr() for tensor in producer[11:14]] == [
        begin,
        begin + rows * 256 * 4,
        begin + rows * 257 * 4,
    ]
    assert [tensor.data_ptr() for tensor in reducer[:3]] == [
        tensor.data_ptr() for tensor in producer[11:14]
    ]
    assert (
        producer[4].data_ptr()
        == reducer[4].data_ptr()
        == (lse.data_ptr() if write_lse else begin + rows * 258 * 4)
    )
    assert producer[17] == splits
    assert (
        reducer[3].data_ptr() == out.data_ptr()
        and reducer[5].data_ptr() == pv.data_ptr()
    )
    with pytest.raises(ValueError, match="workspace_buffer"):
        api._fp8q_split_request_ordered_arguments(
            plan, q, k, v, out, lse, workspace[:-4], order, table, lens, qk, pv
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("batch", [8, 27, 32])
@pytest.mark.parametrize("write_lse", [False, True])
def test_fp8q_split_gpu_public_graph_mutable_metadata(batch, write_lse):
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("requires SM103")
    device = torch.device("cuda")
    torch.manual_seed(4832)
    lengths = ([6, 7, 63, 64, 65, 127, 128, 129, 255, 256, 257] * ((batch + 10) // 11))[
        :batch
    ]
    plan = fp8_plan(lengths, write_lse=write_lse)
    key, value = tuple(
        torch.randn((batch * 8, 64, 2, 256), device=device)
        .mul_(0.1)
        .to(torch.float8_e4m3fn)
        .permute(0, 2, 1, 3)
        for _ in range(2)
    )
    table = torch.arange(batch * 8, dtype=torch.int32, device=device).view(batch, 8)
    lens = torch.tensor(lengths, dtype=torch.int32, device=device)
    order = torch.arange(batch, dtype=torch.int32, device=device)
    qk = torch.tensor([math.log2(math.e) / 16], device=device)
    pv = torch.tensor([1.0], device=device)

    def invoke(q, workspace, out, lse, preparation=None):
        return flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            query=q,
            kv_cache=(key, value),
            workspace_buffer=workspace,
            out=out,
            lse=lse,
            block_tables=table,
            seq_lens=lens,
            max_seq_len=257,
            bmm1_scale_log2=qk,
            bmm2_scale=pv,
            backend="cake",
            enable_pdl=True,
            q_len_per_req=6,
            request_order=order,
            request_order_plan=plan,
            request_order_capture=preparation,
        )

    def reference(q):
        rows = []
        lse_rows = []
        for b, length in enumerate(lengths):
            indices = table[b].long()
            k = (
                key[indices]
                .float()
                .permute(0, 2, 1, 3)
                .reshape(-1, 2, 256)[:length]
                .repeat_interleave(16, dim=1)
            )
            v = (
                value[indices]
                .float()
                .permute(0, 2, 1, 3)
                .reshape(-1, 2, 256)[:length]
                .repeat_interleave(16, dim=1)
            )
            scores = (
                torch.einsum("qhd,khd->hqk", q[b * 6 : (b + 1) * 6].float(), k)
                * qk
                / math.log2(math.e)
            )
            mask = torch.arange(length, device=device)[None, :] > (
                torch.arange(6, device=device)[:, None] + length - 6
            )
            scores.masked_fill_(mask[None], -float("inf"))
            lse_rows.append(scores.logsumexp(-1).transpose(0, 1) * math.log2(math.e))
            rows.append(torch.einsum("hqk,khd->qhd", scores.softmax(-1), v) * pv)
        return torch.cat(rows).to(torch.bfloat16), torch.cat(lse_rows)

    def check(q, out, lse):
        expected_o, expected_lse = reference(q)
        torch.testing.assert_close(out, expected_o, atol=0.1, rtol=0.1)
        if write_lse:
            torch.testing.assert_close(lse, expected_lse, atol=0.01, rtol=0.01)

    instances = []
    for _ in range(8):
        q = (
            torch.randn((batch * 6, 32, 256), device=device)
            .mul_(0.1)
            .to(torch.float8_e4m3fn)
        )
        workspace = torch.empty(
            api._fp8q_split_workspace_bytes(plan), dtype=torch.uint8, device=device
        )
        out = torch.empty(q.shape, dtype=torch.bfloat16, device=device)
        lse = torch.empty(q.shape[:-1], device=device) if write_lse else None
        assert invoke(q, workspace, out, lse).dtype == torch.bfloat16
        check(q, out, lse)
        preparation = api.CakeFmhaRequestOrderedCapture([plan])
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                invoke(q, workspace, out, lse, preparation)
            preparation.finalize()
        except BaseException:
            preparation.discard()
            raise
        instances.append((q, workspace, out, lse, preparation, graph))
    for mutation in range(3):
        order.copy_(
            torch.arange(batch - 1, -1, -1, device=device, dtype=torch.int32)
            if mutation % 2 == 0
            else torch.arange(batch, device=device, dtype=torch.int32)
        )
        lengths = lengths[1:] + lengths[:1]
        lens.copy_(torch.tensor(lengths, device=device, dtype=torch.int32))
        table.copy_(table.roll(1, 0))
        qk.mul_(0.95)
        pv.mul_(0.98)
        for q, _workspace, out, lse, _preparation, graph in instances:
            out.zero_()
            graph.replay()
            check(q, out, lse)
