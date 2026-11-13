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

"""Padding exclusion in the specialized DeepSeek and Llama4 routing kernels."""

import pytest
import torch

from flashinfer import autotune
from flashinfer.fused_moe import trtllm_fp8_block_scale_moe

from .test_trtllm_fp8_token_mask import check_real_rows, weights as weights


@pytest.mark.parametrize("method,shared", [(2, 0), (2, 1), (2, 2), (3, 0)])
@pytest.mark.parametrize("num_tokens", [1, 3, 4, 5, 17, 256, 257, 2048])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("enable_pdl", [False, True])
def test_specialized_routing_mask(
    weights,  # noqa: F811
    method,
    shared,
    num_tokens,
    dtype,
    enable_pdl,
):
    """Cross the warp/cluster thresholds and replay changing padding masks."""
    n, experts = num_tokens, 128
    top_k = 1 if method == 3 else 8
    total_k = top_k + shared
    gen = torch.Generator(device="cuda").manual_seed(9002 + n)
    logits = torch.randn(n, experts, device="cuda", dtype=dtype, generator=gen)
    kwargs = dict(
        **{name: tensor[: experts + shared] for name, tensor in weights.items()},
        routing_bias=(
            torch.randn(experts, device="cuda", dtype=dtype, generator=gen)
            if method == 2
            else None
        ),
        hidden_states=(torch.randn(n, 1024, device="cuda", generator=gen) * 0.1).to(
            torch.float8_e4m3fn
        ),
        hidden_states_scale=torch.ones(8, n, device="cuda"),
        num_experts=experts,
        top_k=top_k,
        n_group=8 if method == 2 else None,
        topk_group=4 if method == 2 else None,
        intermediate_size=512,
        local_expert_offset=0,
        local_num_experts=experts,
        routed_scaling_factor=2.5 if method == 2 else 1.0,
        routing_method_type=method,
        num_fused_shared_experts=shared,
        enable_pdl=enable_pdl,
    )
    replay = torch.empty(n, top_k, dtype=torch.int16, device="cuda")
    baseline = trtllm_fp8_block_scale_moe(
        logits, **kwargs, routing_replay_out=replay
    ).clone()
    baseline_ids = replay.clone()
    original_logits = logits.clone()
    mask = torch.zeros(n, dtype=torch.bool, device="cuda")
    trtllm_fp8_block_scale_moe(logits, **kwargs, token_mask=mask)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = trtllm_fp8_block_scale_moe(
            logits, **kwargs, token_mask=mask, routing_replay_out=replay
        )
    row = torch.arange(n, device="cuda")
    for pattern in [row < 0, row % 2 == 0, row >= 0, row % 2 == 1, row < 0]:
        mask.copy_(pattern)
        logits.copy_(original_logits).masked_fill_(mask[:, None], float("nan"))
        replay.fill_(12345)
        graph.replay()
        check_real_rows(actual, baseline, ~mask)
        assert (replay[mask] == -1).all()
        torch.testing.assert_close(replay[~mask], baseline_ids[~mask])
        _, scores, expanded = trtllm_fp8_block_scale_moe(
            logits, **kwargs, token_mask=mask, do_finalize=False
        )
        scores, expanded = scores.view(n, total_k), expanded.view(n, total_k)
        assert (scores[mask] == 0).all()
        assert (expanded[mask] == -1).all()
        assert int((expanded >= 0).sum()) == int((~mask).sum()) * total_k

    if n == 17:
        mask.copy_(row % 2 == 0)
        compact_kwargs = dict(kwargs)
        compact_kwargs["hidden_states"] = (
            kwargs["hidden_states"]
            .view(torch.uint8)[~mask]
            .contiguous()
            .view(torch.float8_e4m3fn)
        )
        compact_kwargs["hidden_states_scale"] = kwargs["hidden_states_scale"][
            :, ~mask
        ].contiguous()
        compact = trtllm_fp8_block_scale_moe(
            original_logits[~mask].contiguous(), **compact_kwargs
        )
        actual = trtllm_fp8_block_scale_moe(original_logits, **kwargs, token_mask=mask)
        torch.testing.assert_close(actual[~mask], compact, rtol=0.02, atol=0.02)
        if dtype == torch.float32 and not enable_pdl:
            with autotune():
                tuned = trtllm_fp8_block_scale_moe(
                    original_logits, **kwargs, token_mask=mask
                )
            torch.testing.assert_close(tuned[~mask], compact, rtol=0.02, atol=0.02)
