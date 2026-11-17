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

"""Padding masks for logits-routed BF16 and NVFP4 MoE on Blackwell."""

import pytest
import torch

from flashinfer.autotuner import autotune
from flashinfer.fused_moe import (
    WeightLayout,
    convert_to_block_layout,
    trtllm_bf16_moe,
    trtllm_fp4_block_scale_moe,
)
from flashinfer.tllm_enums import SfLayout
from flashinfer.utils import get_compute_capability


@pytest.fixture(autouse=True)
def _blackwell():
    if (
        not torch.cuda.is_available()
        or get_compute_capability(torch.device("cuda"))[0] != 10
    ):
        pytest.skip("Requires an SM100-family GPU")


@pytest.fixture(params=[False, True])
def enable_pdl(request):
    return request.param


@pytest.fixture(params=["bf16", "nvfp4"])
def case(request, enable_pdl):
    torch.manual_seed(42)
    n, h, inter, experts, k = 17, 1024, 1024, 16, 4
    logits = torch.randn(n, experts, device="cuda", dtype=torch.float32)
    common = dict(
        routing_logits=logits,
        routing_bias=None,
        num_experts=experts,
        top_k=k,
        n_group=None,
        topk_group=None,
        intermediate_size=inter,
        local_expert_offset=0,
        local_num_experts=experts,
        routing_method_type=1,
        routed_scaling_factor=None,
        tune_max_num_tokens=n,
        enable_pdl=enable_pdl,
    )
    if request.param == "bf16":

        def weight(shape):
            w = (torch.randn(shape, device="cuda") * 0.02).bfloat16()
            return torch.stack(
                [convert_to_block_layout(e.view(torch.uint8), 128) for e in w]
            ).view(torch.bfloat16)

        common.update(
            hidden_states=(torch.randn(n, h, device="cuda") * 0.1).bfloat16(),
            gemm1_weights=weight((experts, 2 * inter, h)),
            gemm2_weights=weight((experts, h, inter)),
            weight_layout=WeightLayout.BlockMajorK,
        )
        api = trtllm_bf16_moe
    else:

        def packed(shape):
            lo = torch.randint(0, 4, shape, device="cuda", dtype=torch.uint8)
            hi = torch.randint(0, 4, shape, device="cuda", dtype=torch.uint8)
            return lo | (hi << 4)

        def scales(shape):
            return torch.ones(shape, device="cuda").to(torch.float8_e4m3fn)

        scalar = torch.full((experts,), 0.01, device="cuda")
        common.update(
            hidden_states=packed((n, h // 2)),
            hidden_states_scale=scales((n, h // 16)),
            gemm1_weights=packed((experts, 2 * inter, h // 2)),
            gemm1_weights_scale=scales((experts, 2 * inter, h // 16)),
            gemm1_bias=None,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            gemm2_weights=packed((experts, h, inter // 2)),
            gemm2_weights_scale=scales((experts, h, inter // 16)),
            gemm2_bias=None,
            output1_scale_scalar=scalar,
            output1_scale_gate_scalar=scalar,
            output2_scale_scalar=scalar,
            hidden_states_scale_layout=SfLayout.layout_linear,
        )
        api = trtllm_fp4_block_scale_moe

    def launch(mask=None, replay=None):
        result = api(**common, token_mask=mask, routing_replay_out=replay)
        return result if isinstance(result, torch.Tensor) else result[0]

    return launch, n, k


@pytest.mark.parametrize("pattern", ["none", "all_valid", "alternating", "all_padded"])
def test_token_mask(case, pattern):
    launch, n, k = case
    expected = launch().clone()
    mask = (
        None if pattern == "none" else torch.zeros(n, device="cuda", dtype=torch.bool)
    )
    if pattern == "alternating":
        mask[::2] = True
    elif pattern == "all_padded":
        mask.fill_(True)
    replay = torch.full((n, k), -7, dtype=torch.int16, device="cuda")
    actual = launch(mask, replay)
    valid = torch.ones(n, device="cuda", dtype=torch.bool) if mask is None else ~mask
    assert torch.isfinite(actual[valid]).all()
    torch.testing.assert_close(actual[valid], expected[valid], rtol=0.02, atol=0.02)
    if mask is not None:
        assert (replay[mask] == -1).all()
    assert (replay[valid] >= 0).all()


def test_token_mask_graph_replay(case):
    launch, n, k = case
    expected = launch().clone()
    mask = torch.zeros(n, device="cuda", dtype=torch.bool)
    replay = torch.empty((n, k), dtype=torch.int16, device="cuda")
    # Warm up the exact signature before capture.
    launch(mask, replay)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = launch(mask, replay)
    for pattern in ("alternating", "all_padded", "all_valid"):
        mask.fill_(pattern == "all_padded")
        if pattern == "alternating":
            mask[::2] = True
        graph.replay()
        torch.cuda.synchronize()
        assert (replay[mask] == -1).all()
        assert torch.isfinite(actual[~mask]).all()
        torch.testing.assert_close(actual[~mask], expected[~mask], rtol=0.02, atol=0.02)


@pytest.mark.parametrize("invalid", ["dtype", "device", "shape", "strided"])
def test_invalid_token_mask(case, invalid):
    launch, n, _ = case
    launch()  # Compile once before checking native validation.
    mask = torch.zeros(n, device="cuda", dtype=torch.bool)
    if invalid == "dtype":
        mask = mask.int()
    elif invalid == "device":
        mask = mask.cpu()
    elif invalid == "shape":
        mask = mask[:-1]
    else:
        mask = torch.zeros(2 * n, device="cuda", dtype=torch.bool)[::2]
    with pytest.raises((ValueError, RuntimeError, AssertionError)):
        launch(mask)


def test_token_mask_autotune(case):
    launch, n, k = case
    expected = launch().clone()
    mask = torch.zeros(n, device="cuda", dtype=torch.bool)
    mask[::2] = True
    replay = torch.empty((n, k), dtype=torch.int16, device="cuda")
    with autotune():
        actual = launch(mask, replay)
    assert (replay[mask] == -1).all()
    torch.testing.assert_close(actual[~mask], expected[~mask], rtol=0.02, atol=0.02)
    mask.zero_()
    actual = launch(mask, replay)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
