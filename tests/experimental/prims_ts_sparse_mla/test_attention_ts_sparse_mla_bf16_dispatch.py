# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

import pytest
import torch
from tests.experimental.prims_ts_sparse_mla.sparse_mla_test_utils import prepare_fixture

pytest.importorskip("cutlass", minversion="4.7.0")
from flashinfer.attention.prims_ts import BatchSparseMLADecodePagedTSWrapper
from flashinfer.testing.sparse_mla import sparse_mla_reference
from tests.experimental.prims_ts_sparse_mla.test_attention_ts_sparse_mla_coalesced import (
    exercise_graph_routes as _exercise_routes,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="Requires SM100/SM103",
)


@pytest.mark.parametrize(
    "heads,slots",
    [
        (16, 128),
        (16, 129),
        (16, 512),
        (24, 512),
        (32, 512),
        (64, 512),
        (128, 1024),
        (16, 8192),
    ],
)
def test_auto_bf16_direct_graph_routes(heads, slots):
    # Exercise the public default, including the former 256-slot cliff,
    # partial head tiles, wider heads, live holes, and poisoned page padding.
    _exercise_routes(
        torch.bfloat16,
        4,
        True,
        direct=True,
        heads=heads,
        compressed_slots=slots,
        auto_dispatch=True,
    )


@pytest.mark.parametrize("separate_unit_scales", [False, True])
def test_auto_bf16_packed_graph_with_live_scalars(separate_unit_scales):
    torch.manual_seed(83)
    q = (torch.randn(3, 16, 512, device="cuda") * 0.1).to(torch.bfloat16)
    swa = (torch.randn(4, 256, 512, device=q.device) * 0.1).to(q.dtype)
    compressed = (torch.randn(768, 1, 512, device=q.device) * 0.1).to(q.dtype)
    si = torch.arange(128, device=q.device, dtype=torch.int32).repeat(3, 1)
    ci = ((torch.arange(512, device=q.device, dtype=torch.int32) * 13) % 768).repeat(
        3, 1
    )
    si[:, 11::13] = -1
    ci[:, 5::17] = -1
    sl = torch.tensor([128, 99, 0], dtype=torch.int32, device=q.device)
    cl = torch.tensor([512, 409, 0], dtype=torch.int32, device=q.device)
    indptr = torch.tensor([0, 1, 3], dtype=torch.int32, device=q.device)
    sinks = torch.randn(16, device=q.device)
    scale = torch.tensor(512**-0.5, device=q.device)
    output_scale = torch.tensor(0.75, device=q.device)
    unit_scale = torch.ones((), device=q.device)
    extra_scale = unit_scale.clone() if separate_unit_scales else unit_scale
    wrapper = BatchSparseMLADecodePagedTSWrapper()
    wrapper.plan(
        q.device,
        2,
        16,
        max_seq_len_q=2,
        packed_query=True,
        q_data_type=q.dtype,
        max_topk=128,
        max_extra_topk=512,
        has_sinks=True,
        return_lse=True,
    )
    assert wrapper._impl._state["direct_inputs"]
    out = torch.empty_like(q)
    lse = torch.empty(q.shape[:-1], device=q.device)
    kwargs = dict(
        swa_indices=si,
        compressed_indices=ci,
        swa_topk_lens=sl,
        compressed_topk_lens=cl,
        qo_indptr=indptr,
        sinks=sinks,
        softmax_scale=scale,
        output_scale=output_scale,
        q_scale=unit_scale,
        swa_kv_scale=unit_scale,
        compressed_kv_scale=extra_scale,
        out=out,
        lse=lse,
    )

    def check():
        expected, expected_lse = sparse_mla_reference(
            q,
            swa,
            compressed,
            si,
            ci,
            swa_topk_lens=sl,
            compressed_topk_lens=cl,
            sinks=sinks,
            softmax_scale=scale.item(),
            output_scale=output_scale.item(),
        )
        torch.testing.assert_close(out.double(), expected, atol=1e-3, rtol=0.02)
        torch.testing.assert_close(lse.double(), expected_lse, atol=2e-4, rtol=1e-4)

    wrapper.run(**prepare_fixture(wrapper, q, swa, compressed, **kwargs))
    assert wrapper._impl._state["last_direct_inputs"] == (not separate_unit_scales)
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(
            **prepare_fixture(wrapper, q, swa, compressed, validate=False, **kwargs)
        )
    sl.copy_(torch.tensor([0, 128, 31], device=q.device))
    cl.copy_(torch.tensor([0, 257, 63], device=q.device))
    si[1, 17::9] = -1
    ci.copy_(ci.flip(-1))
    scale.mul_(1.3)
    output_scale.fill_(0.5)
    graph.replay()
    check()
