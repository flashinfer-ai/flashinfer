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

"""Padding exclusion in block-score and non-cooperative permutation dispatch."""

import os

import pytest
import torch

from flashinfer.fused_moe import trtllm_fp8_block_scale_moe

from .test_trtllm_fp8_token_mask import check_real_rows, weights as weights


@pytest.mark.parametrize("enable_pdl", [False, True])
@pytest.mark.parametrize("dispatch", ["block_scores", "histogram_offsets"])
def test_token_mask_dispatch(weights, dispatch, enable_pdl):  # noqa: F811
    if os.getenv("FLASHINFER_ROUTING_FORCE_BLOCK_PER_TOKEN", "auto").lower() not in (
        "",
        "auto",
    ):
        pytest.skip("requires the default routing dispatch heuristic")

    local_experts = 8
    top_k = 8
    if dispatch == "block_scores":
        # E >= 1024 selects block-per-token scores outside single-cluster dispatch.
        experts, n = 1024, 257
    else:
        experts = 128
        sm_count = torch.cuda.get_device_properties("cuda").multi_processor_count
        reserved = int(os.getenv("FLASHINFER_TRTLLM_MOE_OVERLAP_RESERVED_SMS", "8"))
        assert 0 <= reserved < sm_count
        # routingCustom::run: maxTokensCoop = moeSms * numThreadsHist * 64 / K.
        # For E=128, getMaxNumExperts(E) is 128. One extra row forces the
        # separate histogram and offsets kernels, regardless of live-token count.
        n = (sm_count - reserved) * 128 * 64 // top_k + 1

    gen = torch.Generator(device="cuda").manual_seed(9010)
    live_rows = torch.tensor([1, 3, n // 3, n // 2, n - 3, n - 1], device="cuda")
    live_logits = torch.randn(6, experts, device="cuda", generator=gen)
    # Give every live token local work while retaining nontrivial routing weights.
    live_logits[:, :local_experts] += 10
    live_hidden = (torch.randn(6, 1024, device="cuda", generator=gen) * 0.1).to(
        torch.float8_e4m3fn
    )
    kwargs = dict(
        **{name: tensor[:local_experts] for name, tensor in weights.items()},
        routing_bias=None,
        num_experts=experts,
        top_k=top_k,
        n_group=None,
        topk_group=None,
        intermediate_size=512,
        local_expert_offset=0,
        local_num_experts=local_experts,
        routed_scaling_factor=1.0,
        routing_method_type=1,
        enable_pdl=enable_pdl,
    )
    reference = trtllm_fp8_block_scale_moe(
        live_logits,
        **kwargs,
        hidden_states=live_hidden,
        hidden_states_scale=torch.ones(8, 6, device="cuda"),
    ).clone()
    logits = torch.full((n, experts), float("nan"), device="cuda")
    logits[live_rows] = live_logits
    hidden = torch.zeros(n, 1024, dtype=torch.float8_e4m3fn, device="cuda")
    hidden.view(torch.uint8)[live_rows] = live_hidden.view(torch.uint8)
    kwargs.update(
        hidden_states=hidden,
        hidden_states_scale=torch.ones(8, n, device="cuda"),
    )
    mask = torch.ones(n, dtype=torch.bool, device="cuda")
    mask[live_rows] = False
    replay = torch.empty(n, top_k, dtype=torch.int16, device="cuda")
    trtllm_fp8_block_scale_moe(logits, **kwargs, token_mask=mask)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = trtllm_fp8_block_scale_moe(
            logits, **kwargs, token_mask=mask, routing_replay_out=replay
        )
    for all_padding in [False, True, False, True]:
        mask.fill_(True)
        if not all_padding:
            mask[live_rows] = False
        replay.fill_(12345)
        graph.replay()
        assert (replay[mask] == -1).all()
        if not all_padding:
            check_real_rows(
                actual[live_rows],
                reference,
                torch.ones(6, dtype=torch.bool, device="cuda"),
            )
            torch.testing.assert_close(
                replay[live_rows].int().sort(-1).values,
                live_logits.topk(top_k, dim=-1).indices.int().sort(-1).values,
            )
        _, scores, expanded = trtllm_fp8_block_scale_moe(
            logits, **kwargs, token_mask=mask, do_finalize=False
        )
        scores, expanded = scores.view(n, top_k), expanded.view(n, top_k)
        assert (scores[mask] == 0).all()
        assert (expanded[mask] == -1).all()
        assert int((expanded >= 0).sum()) == (0 if all_padding else 6 * top_k)
