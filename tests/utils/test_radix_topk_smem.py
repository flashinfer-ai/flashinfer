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
@pytest.mark.parametrize(
    "op,deterministic", [("renorm", False), ("renorm", True), ("mask", False)]
)
@pytest.mark.parametrize("ctas", [1, 5])
@pytest.mark.parametrize("per_row_k", [False, True])
@pytest.mark.parametrize("cuda_graph", [False, True])
def test_radix_top_k_static_shared_memory(
    dtype, op, deterministic, ctas, per_row_k, cuda_graph
):
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    if properties.multi_processor_count <= 16:
        pytest.skip("multi-CTA radix top-k requires more than 16 SMs")

    # Center a small sweep on the old dynamic budget. Unlike a single exact
    # boundary, this also catches nearby failures if the fixed layout changes.
    fixed_smem = 2080
    item_size = torch.empty((), dtype=dtype).element_size()
    chunk_size = (properties.shared_memory_per_block_optin - fixed_smem) // item_size
    for delta in range(-256, 257, 32):
        vocab_size = ctas * (chunk_size + delta) - 1
        _check_radix_boundary(
            dtype, op, deterministic, per_row_k, cuda_graph, device, vocab_size
        )


def _check_radix_boundary(
    dtype, op, deterministic, per_row_k, cuda_graph, device, vocab_size
):
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

        def run(inputs, k):
            return flashinfer.sampling.top_k_renorm_probs(
                inputs, k, is_deterministic=deterministic
            )
    else:
        expected = torch.where(keep, reference_input, float("-inf"))
        run = flashinfer.sampling.top_k_mask_logits

    atol, rtol = (1e-5, 1e-4) if dtype == torch.float32 else (2e-3, 1e-2)

    # Warm up outside capture so the JIT module and workspace are already ready.
    for _ in range(3):
        actual = run(inputs, k)
    torch.cuda.synchronize(device)
    if deterministic:
        torch.testing.assert_close(run(inputs, k), actual, atol=0, rtol=0)
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("op", ["basic", "page_table", "ragged"])
@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("cuda_graph", [False, True])
def test_radix_top_k_transform_shared_memory(
    monkeypatch, dtype, op, deterministic, cuda_graph
):
    monkeypatch.setenv("FLASHINFER_TOPK_ALGO", "multi_cta")
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    if properties.multi_processor_count <= 16:
        pytest.skip("multi-CTA radix top-k requires more than 16 SMs")
    item_size = torch.empty((), dtype=dtype).element_size()
    capacity = (properties.shared_memory_per_block_optin - 2080) // item_size
    for delta in (-32, 0, 32):
        size = 5 * (capacity + delta) - 1
        scores = torch.rand((4, size), device=device, dtype=dtype)
        scores[:, :10] = torch.arange(2, 12, device=device, dtype=dtype)
        lengths = torch.full((4,), size, device=device, dtype=torch.int32)
        offsets = torch.arange(4, device=device, dtype=torch.int32) * size
        page_table = (
            torch.arange(size, device=device, dtype=torch.int32)
            .expand(4, -1)
            .contiguous()
        )

        def run():
            if op == "basic":
                return flashinfer.top_k(scores, 10, deterministic=deterministic)[1]
            if op == "page_table":
                return flashinfer.top_k_page_table_transform(
                    scores, page_table, lengths, 10, deterministic=deterministic
                )
            return flashinfer.top_k_ragged_transform(
                scores, offsets, lengths, 10, deterministic=deterministic
            )

        expected = torch.arange(
            10, device=device, dtype=torch.int64 if op == "basic" else torch.int32
        ).expand(4, -1)
        if op == "ragged":
            expected = expected + offsets[:, None]
        actual = run()
        torch.cuda.synchronize(device)
        if deterministic:
            torch.testing.assert_close(run(), actual, atol=0, rtol=0)
        if cuda_graph:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = run()
            for _ in range(3):
                graph.replay()
                torch.cuda.synchronize(device)
                torch.testing.assert_close(
                    actual.sort(-1).values, expected, atol=0, rtol=0
                )
        else:
            torch.testing.assert_close(actual.sort(-1).values, expected, atol=0, rtol=0)
