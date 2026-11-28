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

import pytest
import torch

import flashinfer


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("heads", [1, 4])
@pytest.mark.parametrize("mode", ["auto", "disabled", "fixed_late", "fixed_none"])
def test_merge_indptr_planning(dtype, index_dtype, heads, mode):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(42)
    q_ptr = torch.tensor([0, 17, 50], dtype=index_dtype)
    kv_ptr = torch.tensor([0, 64, 4160], dtype=index_dtype)
    q = torch.randn(50, heads, 128, device="cuda", dtype=dtype)
    k = torch.randn(4160, 1, 128, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    workspace = torch.empty(64 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace,
        backend="fa2",
    )
    options = {}
    if mode == "disabled":
        options["disable_split_kv"] = True
    elif mode == "fixed_late":
        # Request zero fits one chunk; only request one activates splitting.
        options["fixed_split_size"] = 128
    elif mode == "fixed_none":
        options["fixed_split_size"] = 8192
    wrapper.plan(
        q_ptr,
        kv_ptr,
        heads,
        1,
        128,
        q_data_type=dtype,
        **options,
    )
    if mode != "auto":
        assert bool(wrapper._plan_info[-1]) == (mode == "fixed_late")
    for scale in (1.0, 0.5):
        q.mul_(scale)
        wrapper.plan(
            q_ptr,
            kv_ptr,
            heads,
            1,
            128,
            q_data_type=dtype,
            **options,
        )
        actual = wrapper.run(q, k, v)
        for request in range(2):
            lo, hi = q_ptr[request : request + 2].tolist()
            start, end = kv_ptr[request : request + 2].tolist()
            query = q[lo:hi].float().transpose(0, 1)
            key = k[start:end].float().transpose(0, 1)
            value = v[start:end].float().transpose(0, 1)
            scores = query @ key.transpose(-1, -2) / (128**0.5)
            expected = (scores.softmax(-1) @ value).transpose(0, 1)
            torch.testing.assert_close(
                actual[lo:hi].float(),
                expected,
                atol=0.005,
                rtol=0.02,
            )


def test_no_split_graph_replan():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(13)
    q = torch.randn(50, 1, 128, device="cuda", dtype=torch.float16)
    k = torch.randn(4160, 1, 128, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    q_ptr = torch.tensor([0, 17, 50], dtype=torch.int32)
    kv_ptr = torch.tensor([0, 64, 4160], dtype=torch.int32)
    workspace = torch.empty(64 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace,
        backend="fa2",
        use_cuda_graph=True,
        qo_indptr_buf=torch.empty(3, device="cuda", dtype=torch.int32),
        kv_indptr_buf=torch.empty(3, device="cuda", dtype=torch.int32),
    )

    def plan():
        wrapper.plan(
            q_ptr,
            kv_ptr,
            1,
            1,
            128,
            q_data_type=torch.float16,
            disable_split_kv=True,
        )

    plan()
    assert not bool(wrapper._plan_info[-1])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        wrapper.run(q, k, v)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = wrapper.run(q, k, v)
    for bounds in ([0, 64, 4160], [0, 128, 2048], [0, 64, 4160]):
        kv_ptr.copy_(torch.tensor(bounds, dtype=torch.int32))
        q.mul_(0.9)
        plan()
        graph.replay()
        torch.cuda.synchronize()
        for request in range(2):
            lo, hi = q_ptr[request : request + 2].tolist()
            start, end = bounds[request : request + 2]
            scores = q[lo:hi, 0].float() @ k[start:end, 0].float().T
            expected = (scores / (128**0.5)).softmax(-1) @ v[start:end, 0].float()
            torch.testing.assert_close(
                output[lo:hi, 0].float(),
                expected,
                atol=0.005,
                rtol=0.02,
            )
