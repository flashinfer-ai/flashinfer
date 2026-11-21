"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import math

import pytest
import torch

import flashinfer


def _decode_case(dtype, use_cuda_graph):
    batch, heads, head_dim = 2, 4, 64
    q = torch.zeros(batch, heads, head_dim, device="cuda", dtype=dtype)
    kv = torch.zeros(batch, 2, 1, heads, head_dim, device="cuda", dtype=dtype)
    values = torch.arange(1, batch * heads + 1, device="cuda", dtype=dtype)
    kv[:, 1] = values.reshape(batch, 1, heads, 1)
    # A single zero-logit key gives output == V and LSE == 0 exactly.
    expected = values.reshape(batch, heads, 1).expand_as(q).clone()
    indptr = torch.arange(batch + 1, device="cuda", dtype=torch.int32)
    indices = torch.arange(batch, device="cuda", dtype=torch.int32)
    last_page_len = torch.ones(batch, device="cuda", dtype=torch.int32)
    workspace = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace,
        "NHD",
        use_cuda_graph=use_cuda_graph,
        paged_kv_indptr_buffer=indptr,
        paged_kv_indices_buffer=indices,
        paged_kv_last_page_len_buffer=last_page_len,
        use_tensor_cores=False,
        backend="fa2",
    )
    wrapper.plan(
        indptr,
        indices,
        last_page_len,
        heads,
        heads,
        head_dim,
        1,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    # DecodePlanInfo ends with enable_cuda_graph and split_kv. Graph plans use
    # the split-KV path even for one key; eager plans use direct output here.
    assert bool(wrapper._plan_info[-1]) == use_cuda_graph
    return wrapper, q, kv, expected


def _output_buffer(shape, dtype, layout):
    if layout == "padded":
        backing = torch.full(
            (*shape[:-1], shape[-1] + 8), -11, device="cuda", dtype=dtype
        )
        view = backing[..., : shape[-1]]
    elif layout == "transposed":
        backing = torch.full(
            (*shape[:-2], shape[-1], shape[-2]), -11, device="cuda", dtype=dtype
        )
        view = backing.transpose(-1, -2)
    else:
        offset = 16 if layout == "offset" else 0
        backing = torch.full(
            (math.prod(shape) + offset + 16,), -11, device="cuda", dtype=dtype
        )
        view = backing[offset : offset + math.prod(shape)].view(shape)
    return backing, view


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("use_cuda_graph", [False, True])
@pytest.mark.parametrize("layout", ["padded", "transposed"])
@pytest.mark.parametrize(
    "output_name,return_lse", [("out", True), ("lse", True), ("lse", False)]
)
def test_cuda_core_decode_rejects_output_layout(
    dtype, use_cuda_graph, layout, output_name, return_lse
):
    wrapper, q, kv, expected = _decode_case(dtype, use_cuda_graph)
    baseline, baseline_lse = wrapper.run(q, kv, return_lse=True, enable_pdl=False)
    torch.testing.assert_close(baseline, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        baseline_lse, torch.zeros_like(baseline_lse), rtol=0, atol=0
    )

    out_backing, out = _output_buffer(
        q.shape, dtype, layout if output_name == "out" else "contiguous"
    )
    lse_backing, lse = _output_buffer(
        q.shape[:2], torch.float32, layout if output_name == "lse" else "contiguous"
    )
    assert not (out if output_name == "out" else lse).is_contiguous()
    out_before, lse_before = out_backing.clone(), lse_backing.clone()
    with pytest.raises(ValueError, match=f"{output_name} must be contiguous"):
        wrapper.run(q, kv, out=out, lse=lse, return_lse=return_lse, enable_pdl=False)
    torch.cuda.synchronize()
    assert torch.equal(out_backing, out_before)
    assert torch.equal(lse_backing, lse_before)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("use_cuda_graph", [False, True])
@pytest.mark.parametrize("layout", ["contiguous", "offset"])
def test_cuda_core_decode_contiguous_outputs(dtype, use_cuda_graph, layout):
    wrapper, q, kv, expected = _decode_case(dtype, use_cuda_graph)
    out_backing, out = _output_buffer(q.shape, dtype, layout)
    lse_backing, lse = _output_buffer(q.shape[:2], torch.float32, layout)
    assert out.is_contiguous() and lse.is_contiguous()

    def run():
        return wrapper.run(q, kv, out=out, lse=lse, return_lse=True, enable_pdl=False)

    returned_out, returned_lse = run()
    assert returned_out.data_ptr() == out.data_ptr()
    assert returned_lse.data_ptr() == lse.data_ptr()
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    torch.testing.assert_close(lse, torch.zeros_like(lse), rtol=0, atol=0)

    if use_cuda_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        kv[:, 1].add_(8)
        graph.replay()
        torch.testing.assert_close(out, expected + 8, rtol=0, atol=0)
        torch.testing.assert_close(lse, torch.zeros_like(lse), rtol=0, atol=0)

    for backing, view in [(out_backing, out), (lse_backing, lse)]:
        start = view.storage_offset()
        assert torch.all(backing[:start] == -11)
        assert torch.all(backing[start + view.numel() :] == -11)
