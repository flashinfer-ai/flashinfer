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

"""Fixed-size FP8 MoE inputs can exclude padding without host mask readback."""

import pytest
import torch

import flashinfer

from flashinfer.fused_moe import (
    trtllm_fp8_block_scale_moe,
    trtllm_fp8_block_scale_routed_moe,
)


@pytest.fixture(scope="module")
def weights():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100 family required")
    gen = torch.Generator(device="cuda").manual_seed(9000)
    return dict(
        gemm1_weights=(
            torch.randn(256, 1024, 1024, generator=gen, device="cuda") * 0.1
        ).to(torch.float8_e4m3fn),
        gemm1_weights_scale=torch.ones(256, 8, 8, device="cuda"),
        gemm2_weights=(
            torch.randn(256, 1024, 512, generator=gen, device="cuda") * 0.1
        ).to(torch.float8_e4m3fn),
        gemm2_weights_scale=torch.ones(256, 8, 4, device="cuda"),
    )


def check_real_rows(actual, expected, valid):
    actual, expected = actual[valid].float(), expected[valid].float()
    if expected.numel():
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
        assert (actual - expected).norm() / expected.norm().clamp_min(1e-12) < 0.02


@pytest.mark.parametrize(
    "num_tokens,num_experts",
    [(n, 128) for n in [4, 5, 16, 17, 256, 257, 2048]] + [(257, 256)],
)
@pytest.mark.parametrize("enable_pdl", [False, True])
def test_fp8_token_mask(num_tokens, num_experts, enable_pdl, weights):
    """Exercise block/dynamic/cluster/large routing and changing captured masks."""
    gen = torch.Generator(device="cuda").manual_seed(9000 + num_tokens)
    logits = torch.randn(num_tokens, num_experts, device="cuda", generator=gen)
    hidden = (torch.randn(num_tokens, 1024, device="cuda", generator=gen) * 0.1).to(
        torch.float8_e4m3fn
    )
    kwargs = dict(
        **{name: tensor[:num_experts] for name, tensor in weights.items()},
        routing_bias=None,
        hidden_states=hidden,
        hidden_states_scale=torch.ones(8, num_tokens, device="cuda"),
        num_experts=num_experts,
        top_k=4,
        n_group=None,
        topk_group=None,
        intermediate_size=512,
        local_expert_offset=0,
        local_num_experts=num_experts,
        routed_scaling_factor=1.0,
        routing_method_type=1,
        enable_pdl=enable_pdl,
    )
    baseline = trtllm_fp8_block_scale_moe(logits, **kwargs).clone()
    values, ids = logits.topk(4, dim=-1)
    route_weights = values.softmax(-1).to(torch.bfloat16)
    mask = torch.zeros(num_tokens, dtype=torch.bool, device="cuda")
    replay_ids = torch.empty(num_tokens, 4, dtype=torch.int16, device="cuda")
    trtllm_fp8_block_scale_moe(logits, **kwargs, token_mask=mask)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = trtllm_fp8_block_scale_moe(
            logits, **kwargs, token_mask=mask, routing_replay_out=replay_ids
        )
    row = torch.arange(num_tokens, device="cuda")
    original_logits = logits.clone()
    for pattern in [row < 0, row % 2 == 0, row >= 0, row % 2 == 1, row < 0]:
        mask.copy_(pattern)
        logits.copy_(original_logits).masked_fill_(mask[:, None], float("nan"))
        replay_ids.fill_(12345)
        graph.replay()
        valid = ~mask
        check_real_rows(result, baseline, valid)
        assert (replay_ids[mask] == -1).all()
        torch.testing.assert_close(
            replay_ids[valid].int().sort(-1).values, ids[valid].int().sort(-1).values
        )
        # Check scheduling metadata, including zero live assignments for all-padding.
        _, expert_weights, expanded = trtllm_fp8_block_scale_moe(
            logits, **kwargs, token_mask=mask, do_finalize=False
        )
        expanded = expanded.view(num_tokens, 4)
        assert (expanded[mask] == -1).all()
        assert (expert_weights[mask] == 0).all()
        assert int((expanded >= 0).sum()) == int(valid.sum()) * 4
    mask.copy_(row % 2 == 0)
    routed_ids = ids.int().masked_fill(mask[:, None], -1)
    reference = trtllm_fp8_block_scale_routed_moe((routed_ids, route_weights), **kwargs)
    result = trtllm_fp8_block_scale_moe(original_logits, **kwargs, token_mask=mask)
    check_real_rows(result, reference, ~mask)
    if num_experts == 128 and num_tokens == 17 and not enable_pdl:
        with flashinfer.autotune():
            tuned = trtllm_fp8_block_scale_moe(
                original_logits, **kwargs, token_mask=mask
            )
        check_real_rows(tuned, reference, ~mask)
    with pytest.raises(Exception, match="mask"):
        trtllm_fp8_block_scale_moe(
            original_logits,
            **kwargs,
            token_mask=torch.zeros(num_tokens, dtype=torch.bool),
        )


def test_trace_reference_rejects_nonfinite_real_rows():
    from flashinfer.trace.templates.moe import _fp8_moe_run_experts

    hidden = torch.zeros(2, 128)
    hidden[0, 0] = float("nan")
    args = (
        hidden,
        torch.ones(1, 2),
        torch.zeros(1, 256, 128),
        torch.ones(1, 2, 1),
        torch.zeros(1, 128, 128),
        torch.ones(1, 1, 1),
        torch.ones(2, 1),
        torch.zeros(2, 1, dtype=torch.int64),
        0,
        1,
    )
    for mask in [None, torch.tensor([False, True])]:
        with pytest.raises(ValueError, match="real token"):
            _fp8_moe_run_experts(*args, token_mask=mask)
    result = _fp8_moe_run_experts(*args, token_mask=torch.tensor([True, False]))
    assert result[0].isnan().all()
    assert result[1].isfinite().all()
