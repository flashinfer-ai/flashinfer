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

"""Warm public paged calls retain current bindings and previous captures."""

import math

import pytest
import torch

import flashinfer
from flashinfer.cudnn import prefill as cp


@pytest.mark.skipif(not cp.CUDNN_AVAILABLE, reason="requires cuDNN graph support")
@pytest.mark.parametrize("layout", ["HND", "NHD"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_paged_warm_run_bindings_scales_and_retained_capture(layout, dtype):
    if cp.cudnn.backend_version() < 92400:
        pytest.skip("requires cuDNN 9.24 token offsets")
    torch.manual_seed(181)
    hq, hk, dim, page = 8, 2, 128, 16
    q = torch.randn(5, hq, dim, device="cuda", dtype=dtype)
    kv = torch.randn(6, 2, page, hk, dim, device="cuda", dtype=dtype)
    if layout == "HND":
        kv = kv.transpose(2, 3).contiguous()
    k, v = kv[:, 0], kv[:, 1]
    qo = torch.tensor([0, 3, 5], dtype=torch.int32)
    ip = torch.tensor([0, 3, 5], dtype=torch.int32)
    ix = torch.tensor([3, 0, 4, 2, 1], device="cuda", dtype=torch.int32)
    last = torch.tensor([1, 1], dtype=torch.int32)
    workspace = torch.empty(128 << 20, device="cuda", dtype=torch.uint8)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace,
        layout,
        backend="cudnn",
        use_cuda_graph=True,
        qo_indptr_buf=qo.cuda(),
        paged_kv_indptr_buf=ip.cuda(),
        paged_kv_indices_buf=ix.clone(),
        paged_kv_last_page_len_buf=last.cuda(),
    )

    def plan():
        wrapper.plan(
            qo,
            ip,
            ix,
            last,
            hq,
            hk,
            dim,
            page,
            causal=True,
            q_data_type=dtype,
            max_token_per_sequence=3,
            max_sequence_kv=48,
        )
        assert wrapper._cudnn_plan is not None

    def reference(query, key, value, scales=(1.0, 1.0, 1.0)):
        if layout == "HND":
            key, value = key.transpose(1, 2), value.transpose(1, 2)
        outputs, stats = [], []
        for b in range(2):
            ids = ix[ip[b] : ip[b + 1]].long()
            length = int((ip[b + 1] - ip[b] - 1) * page + last[b])
            kk = key[ids].flatten(0, 1)[:length].double().repeat_interleave(hq // hk, 1)
            vv = (
                value[ids]
                .flatten(0, 1)[:length]
                .double()
                .repeat_interleave(hq // hk, 1)
            )
            qq = query[qo[b] : qo[b + 1]].double()
            scores = torch.einsum("qhd,khd->hqk", qq, kk)
            scores *= dim**-0.5 * scales[0] * scales[1]
            rows = torch.arange(len(qq), device=query.device)
            columns = torch.arange(length, device=query.device)
            scores.masked_fill_(
                columns[None, :] > rows[:, None] + length - len(qq), -torch.inf
            )
            outputs.append(
                torch.einsum("hqk,khd->qhd", scores.softmax(-1), vv) * scales[2]
            )
            stats.append(scores.logsumexp(-1).T)
        return torch.cat(outputs).float(), torch.cat(stats).float()

    def check(output, query, key, value, scales=(1.0, 1.0, 1.0), stats=None):
        expected, lse = reference(query, key, value, scales)
        torch.testing.assert_close(output.float(), expected, atol=0.015, rtol=0.015)
        if stats is not None:
            torch.testing.assert_close(
                stats.T, lse * math.log2(math.e), atol=0.003, rtol=0.003
            )

    plan()
    out = torch.empty_like(q)
    for _ in range(3):
        wrapper.run(q, (k, v), out=out)
    check(out, q, k, v)
    original = torch.cuda.CUDAGraph()
    with torch.cuda.graph(original):
        wrapper.run(q, (k, v), out=out)

    # A current call must use new pointers, even while an old capture remains live.
    fresh_q, fresh_k, fresh_v = -q, k * 0.5, -v * 0.75
    fresh_out = torch.empty_like(out)
    new_workspace = torch.empty_like(workspace)
    wrapper.reset_workspace_buffer(new_workspace, wrapper._int_workspace_buffer)
    wrapper.run(fresh_q, (fresh_k, fresh_v), out=fresh_out)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        wrapper.run(fresh_q, (fresh_k, fresh_v), out=fresh_out)
    torch.cuda.current_stream().wait_stream(stream)
    check(fresh_out, fresh_q, fresh_k, fresh_v)

    # Runtime scales and HN/log2 Stats must still use the complete path.
    scale_values = (0.5, 0.75, 0.25)
    scales = {
        name: torch.tensor([value], device="cuda", dtype=torch.float32)
        for name, value in zip(
            ("q_scale", "k_scale", "v_scale"), scale_values, strict=True
        )
    }
    stats = torch.empty(hq, len(q), device="cuda", dtype=torch.float32)

    def scaled():
        return wrapper.run(
            fresh_q,
            (fresh_k, fresh_v),
            out=fresh_out,
            lse=stats,
            return_lse=True,
            lse_layout="HN",
            lse_base="log2",
            **scales,
        )

    scaled()
    check(fresh_out, fresh_q, fresh_k, fresh_v, scale_values, stats)
    scaled_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(scaled_graph):
        scaled()
    try:
        scale_values = (0.75, 0.5, 0.5)
        for tensor, value in zip(scales.values(), scale_values, strict=True):
            tensor.fill_(value)
        fresh_out.fill_(float("nan"))
        stats.fill_(float("nan"))
        scaled_graph.replay()
        check(fresh_out, fresh_q, fresh_k, fresh_v, scale_values, stats)

        # The retained unscaled capture must observe new metadata and its own data.
        qo = torch.tensor([0, 2, 5], dtype=torch.int32)
        ix = torch.tensor([5, 2, 4, 3, 1], device="cuda", dtype=torch.int32)
        last = torch.tensor([4, 7], dtype=torch.int32)
        plan()
        q.mul_(0.5)
        kv.mul_(-0.625)
        out.fill_(float("nan"))
        original.replay()
        check(out, q, k, v)
        wrapper.run(fresh_q, (fresh_k, fresh_v), out=fresh_out)
        check(fresh_out, fresh_q, fresh_k, fresh_v)
    finally:
        original.reset()
        scaled_graph.reset()
