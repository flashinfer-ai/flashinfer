"""Public FP8-query request-order planning, zero-copy binding and graph replay."""

import dataclasses
import math

import flashinfer
import flashinfer.cake_fmha as api
import pytest
import torch
from flashinfer.jit.cake_fmha_request_ordered import (
    get_cake_fmha_request_ordered_fp8q_manifest,
    get_cake_fmha_request_ordered_manifest,
    get_cake_fmha_request_ordered_module_spec,
)


def fp8_plan(lengths=None, **kwargs):
    return flashinfer.plan_cake_fmha_request_ordered_paged_decode(
        [257] * 64 if lengths is None else lengths,
        6,
        num_q_heads=32,
        num_kv_heads=2,
        query_dtype=torch.float8_e4m3fn,
        **kwargs,
    )


@pytest.mark.parametrize("batch", [64, 128, 160, 192, 224, 256])
def test_fp8q_plan_and_legacy_precision_are_separate(batch):
    plan = fp8_plan([257] * batch)
    manifest = get_cake_fmha_request_ordered_fp8q_manifest()
    assert plan.query_dtype == torch.float8_e4m3fn
    assert plan.module_name == next(
        row["module_name"] for row in manifest["bindings"] if row["batch_size"] == batch
    )
    assert api._is_authenticated_request_ordered_plan(plan)
    assert (
        get_cake_fmha_request_ordered_module_spec(plan.module_name).tma_workspace_bytes
        == 384
    )
    legacy = flashinfer.plan_cake_fmha_request_ordered_paged_decode(
        [257] * 64, 6, num_q_heads=32, num_kv_heads=2
    )
    assert (
        legacy.query_dtype == torch.bfloat16 and legacy.module_name != plan.module_name
    )
    assert (
        get_cake_fmha_request_ordered_manifest(32, 2)["contract"]["query_output_dtype"]
        == "bfloat16"
    )


@pytest.mark.parametrize(
    "change",
    [
        dict(grid=(1, 1, 64)),
        dict(workspace_parts=2),
        dict(total_tiles=64),
        dict(query_dtype=torch.bfloat16),
        dict(module_name="unknown"),
    ],
)
def test_fp8q_forged_plan_is_rejected(change):
    assert not api._is_authenticated_request_ordered_plan(
        dataclasses.replace(fp8_plan(), **change)
    )


@pytest.mark.parametrize(
    "lengths,kwargs",
    [
        ([257] * 32, {}),
        ([257] * 64, {"write_lse": True}),
        ([5] * 64, {}),
        ([257] * 64, {"num_kv_splits": 2}),
    ],
)
def test_fp8q_unexported_shapes_and_modes_are_rejected(lengths, kwargs):
    with pytest.raises(ValueError):
        fp8_plan(lengths, **kwargs)


@pytest.mark.parametrize("batch", [64, 128, 160, 192, 224, 256])
def test_fp8q_binding_preserves_caller_storage(batch):
    q = torch.empty((batch * 6, 32, 256), dtype=torch.float8_e4m3fn)
    k = torch.empty((8, 64, 2, 256), dtype=torch.float8_e4m3fn).permute(0, 2, 1, 3)
    v = torch.empty_like(k)
    out = torch.empty(q.shape, dtype=torch.bfloat16)
    dummy = torch.empty(1, dtype=torch.float32)
    order = torch.arange(batch, dtype=torch.int32)
    table = torch.empty((batch, 8), dtype=torch.int32)
    lens = torch.full((batch,), 257, dtype=torch.int32)
    scales = (torch.tensor([math.log2(math.e) / 16]), torch.tensor([1.0]))
    args = api._fp8q_request_ordered_arguments(
        q, k, v, out, dummy, order, table, lens, *scales
    )
    assert len(args) == 19 and args[0].shape == (batch * 6, 2, 16, 256)
    for view, original in zip(
        (
            args[0],
            args[1],
            args[2],
            args[3],
            args[5],
            args[7],
            args[8],
            args[9],
            args[10],
        ),
        (q, k, v, out, order, table, lens, *scales),
        strict=True,
    ):
        assert view.data_ptr() == original.data_ptr()
    assert args[0].stride() == (8192, 4096, 256, 1)
    assert args[1].stride() == (32768, 256, 512, 1)
    assert args[14:] == (8, 0, 1, 1, 0)
    with pytest.raises(ValueError, match="strides"):
        api._fp8q_request_ordered_arguments(
            q, k.contiguous(), v, out, dummy, order, table, lens, *scales
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("batch", [64, 128, 160, 192, 224, 256])
def test_fp8q_gpu_public_graph_mutable_metadata(batch):
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("requires SM103")
    device = torch.device("cuda")
    torch.manual_seed(4832)
    lengths = ([6, 7, 63, 64, 65, 127, 128, 129, 255, 256, 257] * ((batch + 10) // 11))[
        :batch
    ]
    plan = fp8_plan(lengths)
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

    def invoke(q, workspace, out, preparation=None):
        return flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            query=q,
            kv_cache=(key, value),
            workspace_buffer=workspace,
            out=out,
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
            rows.append(torch.einsum("hqk,khd->qhd", scores.softmax(-1), v) * pv)
        return torch.cat(rows).to(torch.bfloat16)

    instances = []
    for _ in range(8):
        q = (
            torch.randn((batch * 6, 32, 256), device=device)
            .mul_(0.1)
            .to(torch.float8_e4m3fn)
        )
        workspace = torch.empty(512, dtype=torch.uint8, device=device)
        out = torch.empty(q.shape, dtype=torch.bfloat16, device=device)
        assert invoke(q, workspace, out).dtype == torch.bfloat16
        torch.testing.assert_close(out, reference(q), atol=0.1, rtol=0.1)
        preparation = api.CakeFmhaRequestOrderedCapture([plan])
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                invoke(q, workspace, out, preparation)
            preparation.finalize()
        except BaseException:
            preparation.discard()
            raise
        instances.append((q, workspace, out, preparation, graph))
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
        for q, _workspace, out, _preparation, graph in instances:
            out.zero_()
            graph.replay()
            torch.testing.assert_close(out, reference(q), atol=0.1, rtol=0.1)
