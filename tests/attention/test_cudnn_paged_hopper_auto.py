# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""H100 paged auto must preserve native schedules and retained cuDNN graphs."""

from itertools import accumulate

import pytest
import torch

import flashinfer
import flashinfer.prefill as fp


def require_h100():
    if not torch.cuda.is_available():
        pytest.skip("requires H100")
    p = torch.cuda.get_device_properties(0)
    if p.name != "NVIDIA H100 80GB HBM3" or p.multi_processor_count != 132:
        pytest.skip("requires the qualified H100 80GB HBM3 target")
    if not fp._cudnn_supports_paged_auto():
        pytest.skip("requires the qualified cuDNN paged runtime")


@pytest.mark.parametrize("first", ["fa2", "fa3", "cudnn"])
@pytest.mark.parametrize("return_lse", [False, True])
@pytest.mark.parametrize("caller_table", [False, True])
def test_hopper_paged_auto_retained_capture(first, return_lse, caller_table):
    require_h100()
    shapes = [[64, 64, 64, 64], [65, 63, 64, 64]]
    start = 0 if first == "fa2" else 1
    b, hq, hk, d, page, nk = 4, 32, 4, 256, 16, 4096
    pageptr = torch.arange(b + 1, dtype=torch.int32) * (nk // page)
    indices = torch.arange(b * nk // page, dtype=torch.int32, device="cuda")
    table = indices.reshape(b, nk // page).clone()
    last = torch.full((b,), page, dtype=torch.int32)
    lengths = torch.full((b,), nk, dtype=torch.int32)
    qptr = torch.tensor([0, *accumulate(shapes[start])], dtype=torch.int32)
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(256 << 20, device="cuda", dtype=torch.uint8),
        "HND",
        backend="auto",
        use_cuda_graph=True,
        qo_indptr_buf=qptr.cuda(),
        paged_kv_indptr_buf=pageptr.cuda(),
        paged_kv_indices_buf=indices.clone(),
        paged_kv_last_page_len_buf=last.cuda(),
    )
    q = torch.zeros(256, hq, d, dtype=torch.bfloat16, device="cuda")
    kv = torch.zeros(b * nk // page, 2, hk, page, d, dtype=q.dtype, device=q.device)
    k, v = kv[:, 0], kv[:, 1]
    positions = torch.arange(nk, device=q.device).repeat(b).reshape(-1, page)
    source_weights = torch.arange(1, b + 1, device=q.device).repeat_interleave(
        nk // page
    )
    v.copy_(
        (positions >= nk - 64)[:, None, :, None]
        * (128 * source_weights[:, None, None, None])
    )
    out = torch.empty_like(q)
    lse = torch.empty(256, hq, dtype=torch.float32, device=q.device)

    def plan(qs, initial=False):
        qptr.copy_(torch.tensor([0, *accumulate(qs)], dtype=torch.int32))
        # An overdeclared first plan keeps FA3. A later tighter declaration
        # must not abandon the native schedule retained by its capture.
        maximum = 128 if initial and first == "fa3" else max(qs)
        w.plan(
            qptr,
            pageptr,
            indices,
            last,
            hq,
            hk,
            d,
            page,
            causal=True,
            q_data_type=q.dtype,
            seq_lens=lengths,
            block_tables=table if caller_table else None,
            max_token_per_sequence=maximum,
            max_sequence_kv=nk,
        )

    def run():
        return w.run(
            q,
            (k, v),
            out=out,
            lse=lse if return_lse else None,
            return_lse=return_lse,
            lse_base="ln",
        )

    def check(qs, multiplier, rowmap):
        counts = torch.cat(
            [
                torch.arange(nk - nq + 1, nk + 1, device=q.device, dtype=torch.float64)
                for nq in qs
            ]
        )
        weights = torch.cat(
            [
                torch.full((nq,), rowmap[i] + 1, device=q.device, dtype=torch.float64)
                for i, nq in enumerate(qs)
            ]
        )
        expected = (
            128 * multiplier * weights * (counts - (nk - 64)).clamp_min(0) / counts
        ).float()
        torch.testing.assert_close(
            out.float(), expected[:, None, None].expand_as(out), atol=0.02, rtol=0.02
        )
        if return_lse:
            torch.testing.assert_close(
                lse,
                counts.log().float()[:, None].expand_as(lse),
                atol=0.002,
                rtol=0.002,
            )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        plan(shapes[start], initial=True)
        assert w._backend == first
        if first == "cudnn":
            assert w._cudnn_paged_bounds is None
        for _ in range(5):
            run()
        check(shapes[start], 1, list(range(b)))
        with torch.cuda.graph(graph, stream=stream):
            run()
        # Both directions preserve total Q but move the causal boundary.
        for phase, index in enumerate([1 - start, start]):
            multiplier = phase + 2
            v.copy_(
                (positions >= nk - 64)[:, None, :, None]
                * (128 * multiplier * source_weights[:, None, None, None])
            )
            # Distinct source-row values make a stale page table observable.
            rowmap = torch.arange(b).roll(phase + 1).tolist()
            indices.copy_(
                torch.arange(b * nk // page, device=q.device, dtype=torch.int32)
                .reshape(b, -1)
                .roll(phase + 1, 0)
                .flatten()
            )
            table.copy_(indices.reshape_as(table))
            plan(shapes[index])
            expected = "fa2" if first in ("fa2", "cudnn") else "fa3"
            assert w._backend == expected
            out.fill_(float("nan"))
            lse.fill_(float("nan"))
            run()
            check(shapes[index], multiplier, rowmap)
            out.fill_(float("nan"))
            lse.fill_(float("nan"))
            graph.replay()
            check(shapes[index], multiplier, rowmap)
        stream.synchronize()
    graph.reset()


@pytest.mark.parametrize("nq", [64, 65, 129])
@pytest.mark.parametrize("nk", [512, 16384])
def test_hopper_paged_auto_runtime_scales(nq, nk):
    require_h100()
    torch.manual_seed(83)
    hq, hk, d, page = 16, 4, 256, 16
    q = torch.randn(nq, hq, d, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(nk // page, 2, hk, page, d, device=q.device, dtype=q.dtype)
    k, v = kv[:, 0], kv[:, 1]
    ix = torch.randperm(nk // page, device=q.device, dtype=torch.int32)
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(128 << 20, device=q.device, dtype=torch.uint8),
        "HND",
        backend="auto",
    )
    w.plan(
        torch.tensor([0, nq], dtype=torch.int32),
        torch.tensor([0, nk // page], dtype=torch.int32),
        ix,
        torch.tensor([page], dtype=torch.int32),
        hq,
        hk,
        d,
        page,
        causal=True,
        q_data_type=q.dtype,
    )
    assert w._backend == ("fa2" if nq <= 64 else "cudnn" if nk == 16384 else "fa3")
    keys = (
        k[ix.long()]
        .permute(0, 2, 1, 3)
        .reshape(nk, hk, d)
        .double()
        .repeat_interleave(hq // hk, 1)
    )
    values = (
        v[ix.long()]
        .permute(0, 2, 1, 3)
        .reshape(nk, hk, d)
        .double()
        .repeat_interleave(hq // hk, 1)
    )
    allowed = (
        torch.arange(nk, device=q.device)[None, :]
        <= (nk - nq + torch.arange(nq, device=q.device))[:, None]
    )
    for qs, ks, vs in [(0.5, 1.0, 1.0), (1.0, 0.25, 1.0), (1.0, 1.0, 0.75)]:
        scores = torch.einsum("qhd,khd->hqk", q.double(), keys) * (d**-0.5 * qs * ks)
        scores.masked_fill_(~allowed[None], float("-inf"))
        ref = torch.einsum("hqk,khd->qhd", scores.softmax(-1), values) * vs
        stats = scores.logsumexp(-1).transpose(0, 1)
        out, lse = w.run(
            q,
            (k, v),
            return_lse=True,
            lse_base="ln",
            q_scale=qs,
            k_scale=ks,
            v_scale=vs,
        )
        torch.testing.assert_close(out.float(), ref.float(), atol=0.015, rtol=0.015)
        torch.testing.assert_close(lse, stats.float(), atol=0.003, rtol=0.003)


@pytest.mark.parametrize("first", ["fa3", "cudnn"])
@pytest.mark.parametrize("return_lse", [False, True])
def test_hopper_work_boundary_preserves_captures(first, return_lse):
    require_h100()
    nq, hq, hk, d, page, capacity = 65, 32, 8, 256, 16, 8192
    lengths = [8064, 8192]
    start = int(first == "cudnn")
    qptr = torch.tensor([0, nq], dtype=torch.int32)
    indices = torch.arange(capacity // page, dtype=torch.int32, device="cuda")
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(128 << 20, dtype=torch.uint8, device="cuda"),
        "HND",
        backend="auto",
        use_cuda_graph=True,
        qo_indptr_buf=qptr.cuda(),
        paged_kv_indptr_buf=torch.tensor(
            [0, capacity // page], dtype=torch.int32, device="cuda"
        ),
        paged_kv_indices_buf=indices.clone(),
        paged_kv_last_page_len_buf=torch.tensor(
            [page], dtype=torch.int32, device="cuda"
        ),
    )
    q = torch.zeros(nq, hq, d, dtype=torch.bfloat16, device="cuda")
    kv = torch.zeros(capacity // page, 2, hk, page, d, dtype=q.dtype, device=q.device)
    k, v = kv[:, 0], kv[:, 1]
    positions = torch.arange(capacity, device=q.device).reshape(-1, page)
    out = torch.empty_like(q)
    lse = torch.empty(nq, hq, dtype=torch.float32, device=q.device)

    def plan(nk):
        w.plan(
            qptr,
            torch.tensor([0, nk // page], dtype=torch.int32),
            indices[: nk // page],
            torch.tensor([page], dtype=torch.int32),
            hq,
            hk,
            d,
            page,
            causal=True,
            q_data_type=q.dtype,
            seq_lens=torch.tensor([nk], dtype=torch.int32),
            max_token_per_sequence=nq,
            max_sequence_kv=nk,
        )

    def run():
        return w.run(
            q,
            (k, v),
            out=out,
            lse=lse if return_lse else None,
            return_lse=return_lse,
            lse_base="ln",
        )

    def check(nk, scale):
        count = torch.arange(nk - nq + 1, nk + 1, dtype=torch.float64, device=q.device)
        expected = (scale * (count - 8000).clamp_min(0) / count).float()
        torch.testing.assert_close(
            out.float(), expected[:, None, None].expand_as(out), atol=0.002, rtol=0.02
        )
        if return_lse:
            torch.testing.assert_close(
                lse, count.log().float()[:, None].expand_as(lse), atol=0.002, rtol=0.002
            )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        v.copy_((positions >= 8000)[:, None, :, None] * 8)
        plan(lengths[start])
        assert w._backend == first
        for _ in range(5):
            run()
        check(lengths[start], 8)
        with torch.cuda.graph(graph, stream=stream):
            run()
        for phase, nk in enumerate([lengths[1 - start], lengths[start]]):
            scale = 16 * (phase + 1)
            v.copy_((positions >= 8000)[:, None, :, None] * scale)
            plan(nk)
            assert w._backend == "fa3"
            out.fill_(float("nan"))
            lse.fill_(float("nan"))
            run()
            check(nk, scale)
            out.fill_(float("nan"))
            lse.fill_(float("nan"))
            graph.replay()
            check(nk, scale)
        stream.synchronize()
    graph.reset()
