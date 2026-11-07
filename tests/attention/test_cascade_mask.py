"""
Copyright (c) 2023 by FlashInfer team.

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
@pytest.mark.parametrize("use_cuda_graph", [False, True])
@pytest.mark.parametrize("mask_stride", [0, 1, 2, 3])
@pytest.mark.parametrize("pattern", ["all_true", "all_false", "alternating"])
def test_merge_state_in_place_mask_stride(dtype, use_cuda_graph, mask_stride, pattern):
    if mask_stride == 0 and pattern == "alternating":
        pytest.skip("A broadcast mask has the same value in every row")

    seq_len = 17
    # Keep padding opposite to selected entries and use a nonzero storage offset.
    storage = torch.full(
        (1 + seq_len * max(mask_stride, 1),),
        pattern == "all_false",
        dtype=torch.bool,
        device="cuda",
    )
    if mask_stride == 0:
        storage[1] = pattern == "all_true"
        mask = storage[1:2].expand(seq_len)
    else:
        mask = storage[1::mask_stride]
        if pattern == "alternating":
            mask.copy_(torch.arange(seq_len, device="cuda") % 2 == 0)
        else:
            mask.fill_(pattern == "all_true")
    assert mask.stride(0) == mask_stride
    assert mask.storage_offset() == 1

    v = torch.zeros((seq_len, 2, 64), dtype=dtype, device="cuda")
    s = torch.zeros((seq_len, 2), dtype=torch.float32, device="cuda")
    v_other = torch.ones_like(v)
    s_other = torch.zeros_like(s)

    # Warm up the JIT before capture.
    flashinfer.merge_state_in_place(v, s, v_other, s_other, mask)
    v.zero_()
    s.zero_()
    if use_cuda_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            flashinfer.merge_state_in_place(v, s, v_other, s_other, mask)

    for _ in range(2):
        v.zero_()
        s.zero_()
        if use_cuda_graph:
            graph.replay()
        else:
            flashinfer.merge_state_in_place(v, s, v_other, s_other, mask)
        # Equal logsumexp weights average the states and add one in base 2.
        # Unselected rows must remain exactly unchanged.
        expected_v = (mask.to(dtype) * 0.5)[:, None, None].expand_as(v)
        expected_s = mask.to(s.dtype)[:, None].expand_as(s)
        torch.testing.assert_close(v, expected_v, rtol=0, atol=0)
        torch.testing.assert_close(s, expected_s, rtol=0, atol=0)
        # Replays must read the current mask values through the original view.
        storage.logical_not_()
