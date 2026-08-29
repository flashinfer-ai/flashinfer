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

from flashinfer.fused_moe.bgmv_moe import prepare_bgmv_moe


_PERF_SHAPES = [
    (hidden_size, num_tokens, torch.bfloat16)
    for hidden_size in (3072, 2688)
    for num_tokens in (1, 4, 8, 32, 256, 512, 1024)
]
_FP16_SHAPES = [
    (3072, 8, torch.float16),
    (2688, 4, torch.float16),
]


def _require_sm100():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("generated Blackwell BGMV MoE tests require exact SM100")


def _make_inputs(hidden_size, num_tokens, dtype, *, arbitrary_routes=False):
    torch.manual_seed(42)
    device = "cuda"
    rank = 32
    num_experts = 128
    num_loras = 2
    top_k = 2
    num_pairs = num_tokens * top_k
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device) * 0.1
    lora_a = (
        torch.randn(
            num_loras,
            num_experts,
            rank,
            hidden_size,
            dtype=dtype,
            device=device,
        )
        * 0.01
    )
    lora_b = (
        torch.randn(
            num_loras,
            num_experts,
            hidden_size,
            rank,
            dtype=dtype,
            device=device,
        )
        * 0.01
    )
    sorted_token_ids = torch.arange(
        num_tokens, dtype=torch.int64, device=device
    ).repeat_interleave(top_k)
    expert_ids = torch.randint(
        0, num_experts, (num_pairs,), dtype=torch.int64, device=device
    )
    topk_weights = torch.softmax(
        torch.randn(num_tokens, top_k, dtype=torch.float32, device=device), dim=-1
    ).reshape(-1)
    lora_indices = torch.randint(
        0, num_loras, (num_tokens,), dtype=torch.int64, device=device
    )
    if num_tokens > 1:
        lora_indices[0] = -1
    if arbitrary_routes:
        order = torch.randperm(num_pairs, device=device)
        sorted_token_ids = sorted_token_ids[order].contiguous()
        expert_ids = expert_ids[order].contiguous()
        topk_weights = topk_weights[order].contiguous()
    return (
        x,
        [lora_a],
        [lora_b],
        sorted_token_ids,
        expert_ids,
        lora_indices,
        topk_weights,
        num_experts,
    )


def _reference(inputs):
    (
        x,
        lora_a_weights,
        lora_b_weights,
        sorted_token_ids,
        expert_ids,
        lora_indices,
        topk_weights,
        _num_experts,
    ) = inputs
    lora_a = lora_a_weights[0]
    lora_b = lora_b_weights[0]
    num_tokens, hidden_size = x.shape
    output = torch.zeros(num_tokens, hidden_size, dtype=torch.float32, device=x.device)
    valid = (sorted_token_ids >= 0) & (sorted_token_ids < num_tokens)
    valid_pairs = torch.nonzero(valid, as_tuple=False).flatten()
    for start in range(0, valid_pairs.numel(), 64):
        pair_ids = valid_pairs[start : start + 64]
        tokens = sorted_token_ids[pair_ids]
        loras = lora_indices[tokens]
        active = loras >= 0
        if not bool(active.any()):
            continue
        pair_ids = pair_ids[active]
        tokens = tokens[active]
        loras = loras[active]
        experts = expert_ids[pair_ids]
        a = lora_a[loras, experts].float()
        shrink = torch.bmm(x[tokens].float().unsqueeze(1), a.transpose(1, 2)).squeeze(1)
        b = lora_b[loras, experts].float()
        delta = torch.bmm(b, shrink.unsqueeze(2)).squeeze(2)
        delta *= topk_weights[pair_ids].unsqueeze(1)
        output.index_add_(0, tokens, delta)
    return output


@pytest.mark.parametrize(
    ("hidden_size", "num_tokens", "dtype"), _PERF_SHAPES + _FP16_SHAPES
)
def test_prepared_pipeline_matches_reference(hidden_size, num_tokens, dtype):
    _require_sm100()
    inputs = _make_inputs(hidden_size, num_tokens, dtype)
    expected = _reference(inputs)
    plan = prepare_bgmv_moe(*inputs, backend="blackwell")
    actual = plan.run()
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)

    # Replays keep exact pointers but consume current tensor contents.
    inputs[0].mul_(0.75)
    expected_replay = _reference(inputs)
    actual_replay = plan.run()
    torch.testing.assert_close(actual_replay, expected_replay, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_tokens", [4, 32])
def test_arbitrary_route_order_and_nondefault_stream(dtype, num_tokens):
    _require_sm100()
    inputs = _make_inputs(2688, num_tokens, dtype, arbitrary_routes=True)
    expected = _reference(inputs)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        plan = prepare_bgmv_moe(*inputs, backend="blackwell")
        actual = plan.run()
    stream.synchronize()
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_invalid_pair_padding_and_outer_graph_capture(dtype):
    _require_sm100()
    inputs = list(_make_inputs(3072, 8, dtype))
    device = inputs[0].device
    inputs[3] = torch.cat(
        [inputs[3], torch.tensor([-1, 8], dtype=torch.int64, device=device)]
    )
    inputs[4] = torch.cat([inputs[4], torch.zeros(2, dtype=torch.int64, device=device)])
    inputs[6] = torch.cat(
        [inputs[6], torch.zeros(2, dtype=torch.float32, device=device)]
    )
    inputs = tuple(inputs)
    expected = _reference(inputs)
    plan = prepare_bgmv_moe(*inputs, backend="blackwell")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = plan.run()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
