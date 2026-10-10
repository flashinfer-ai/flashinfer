"""Regression tests for static shared memory in radix top-k launch budgeting."""

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

import flashinfer
import pytest
import torch


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("op", ["renorm", "mask"])
@pytest.mark.parametrize("per_row_k", [False, True])
@pytest.mark.parametrize("cuda_graph", [False, True])
def test_radix_top_k_static_shared_memory(dtype, op, per_row_k, cuda_graph):
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    if properties.multi_processor_count <= 16:
        pytest.skip("multi-CTA radix top-k requires more than 16 SMs")

    # The old launcher gave the dynamic ordered-value array all of the remaining
    # opt-in space, without accounting for the kernel's static shared scratch.
    # An odd vocabulary selects VEC_SIZE=1. Five equally sized chunks fill that
    # space on this device, so static scratch makes the unpatched launch fail.
    fixed_smem = 2080  # two 256-bin histograms, five scalars, aligned to 16 B
    item_size = torch.empty((), dtype=dtype).element_size()
    chunk_size = (properties.shared_memory_per_block_optin - fixed_smem) // item_size
    vocab_size = 5 * chunk_size - 1
    assert vocab_size % 2 == 1

    generator = torch.Generator(device=device).manual_seed(42)
    inputs = torch.rand((4, vocab_size), device=device, generator=generator)
    # Make the top ten values distinct even after casting to half precision.
    inputs[:, :10] = torch.arange(2, 12, device=device, dtype=torch.float32)
    if op == "renorm":
        inputs = inputs / inputs.sum(dim=-1, keepdim=True)
    inputs = inputs.to(dtype)
    k_values = torch.tensor([1, 3, 7, 10], device=device, dtype=torch.int32)
    k = k_values if per_row_k else 10

    reference_input = inputs.float()
    sorted_values = reference_input.topk(10, dim=-1).values
    if per_row_k:
        pivot = sorted_values.gather(1, (k_values.long() - 1).unsqueeze(1))
    else:
        pivot = sorted_values[:, -1:]
    keep = reference_input >= pivot
    if op == "renorm":
        expected = torch.where(keep, reference_input, 0)
        expected = expected / expected.sum(dim=-1, keepdim=True)
        run = flashinfer.sampling.top_k_renorm_probs
    else:
        expected = torch.where(keep, reference_input, float("-inf"))
        run = flashinfer.sampling.top_k_mask_logits

    atol, rtol = (1e-5, 1e-4) if dtype == torch.float32 else (2e-3, 1e-2)

    # Warm up outside capture so the JIT module and workspace are already ready.
    for _ in range(3):
        actual = run(inputs, k)
    torch.cuda.synchronize(device)
    if cuda_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = run(inputs, k)
        for _ in range(3):
            graph.replay()
            torch.cuda.synchronize(device)
            torch.testing.assert_close(actual.float(), expected, atol=atol, rtol=rtol)
    else:
        torch.testing.assert_close(actual.float(), expected, atol=atol, rtol=rtol)
    if op == "renorm":
        assert torch.equal(actual > 0, keep)
    else:
        assert torch.equal(torch.isfinite(actual), keep)
