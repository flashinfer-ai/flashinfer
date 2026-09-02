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

Tests for the fused AlphaMoE gating router (``alphamoe_fused_router`` +
``allocate_alphamoe_route_plan``, "vibecuda" backend).

The oracle recomputes the complete block-sparse routing metadata bundle in
plain torch: stable descending top-k over the routed experts (ties keep the
lower expert index), optional shared-expert column, fp32 max-subtracted
softmax over the selected logits, expert histogram, block_m-aligned padded
offsets and extent, scatter offsets equal to the expert counts, expert-
grouped flat route ids with sentinel padding, and per-block expert ids.
"""

import pytest
import torch

from flashinfer.fused_moe import (
    AlphaMoeRoutePlan,
    allocate_alphamoe_route_plan,
    alphamoe_fused_router,
)
from flashinfer.utils import BackendSupportedError


def _router_reference(logits, top_k, block_m, has_shared_expert):
    """Exact torch oracle for the routing bundle (see module docstring)."""
    num_tokens, num_experts = logits.shape
    routed_experts = num_experts - int(has_shared_expert)
    routed_top_k = top_k - int(has_shared_expert)
    order = torch.argsort(
        logits[:, :routed_experts], dim=-1, descending=True, stable=True
    )[:, :routed_top_k]
    selected = torch.gather(logits, 1, order)
    if has_shared_expert:
        shared = torch.full(
            (num_tokens, 1),
            num_experts - 1,
            dtype=torch.int64,
            device=logits.device,
        )
        order = torch.cat((order, shared), dim=-1)
        selected = torch.cat((selected, logits[:, -1:]), dim=-1)
    topk_ids = order.to(torch.int32)
    topk_weights = torch.softmax(selected, dim=-1)

    flat = topk_ids.cpu().reshape(-1).to(torch.int64)
    counts = torch.bincount(flat, minlength=num_experts).to(torch.int32)
    padded = (counts + block_m - 1) // block_m * block_m
    offsets = torch.empty(num_experts + 1, dtype=torch.int32)
    offsets[0] = 0
    offsets[1:] = torch.cumsum(padded, dim=0)
    extent = int(offsets[-1])

    pairs = num_tokens * top_k
    nonempty = min(num_experts, pairs)
    max_blocks = nonempty + (pairs - nonempty) // block_m
    sorted_ids = torch.zeros(max_blocks * block_m, dtype=torch.int32)
    expert_ids = torch.zeros(max_blocks, dtype=torch.int32)
    sentinel = pairs
    for expert in range(num_experts):
        start = int(offsets[expert])
        count = int(counts[expert])
        end = int(offsets[expert + 1])
        if count == 0:
            continue
        routes = torch.nonzero(flat == expert).flatten().to(torch.int32)
        sorted_ids[start : start + count] = routes
        sorted_ids[start + count : end] = sentinel
        expert_ids[start // block_m : end // block_m] = expert

    return (
        topk_weights.cpu(),
        topk_ids.cpu(),
        sorted_ids,
        expert_ids,
        torch.tensor([extent], dtype=torch.int32),
        counts,
        offsets,
        counts.clone(),
    )


def _max_blocks(num_tokens, num_experts, top_k, block_m):
    pairs = num_tokens * top_k
    nonempty = min(num_experts, pairs)
    return nonempty + (pairs - nonempty) // block_m


def _assert_bundle(out, ref):
    names = (
        "topk_weights",
        "topk_ids",
        "sorted_token_ids",
        "expert_ids",
        "num_tokens_post_padded",
        "expert_counts",
        "expert_offsets",
        "expert_scatter_offsets",
    )
    assert len(out) == len(names)
    for name, actual, expected in zip(names, out, ref):
        if name == "topk_weights":
            torch.testing.assert_close(
                actual.cpu(), expected, rtol=3e-4, atol=3e-4, msg=name
            )
        else:
            torch.testing.assert_close(
                actual.cpu(), expected, rtol=0, atol=0, msg=name
            )


@pytest.mark.parametrize(
    "num_tokens,num_experts,top_k,block_m,has_shared_expert",
    [
        (1, 512, 2, 16, True),
        (8, 32, 4, 8, False),
        (8, 257, 9, 8, True),
        (32, 512, 8, 16, False),
        (128, 512, 8, 16, False),
        (33, 65, 5, 4, True),
        (256, 128, 8, 32, False),
    ],
)
def test_alphamoe_fused_router_correctness(
    num_tokens, num_experts, top_k, block_m, has_shared_expert
):
    torch.manual_seed(num_tokens * 1000 + num_experts + top_k)
    logits = torch.randn(num_tokens, num_experts, dtype=torch.float32, device="cuda")
    ref = _router_reference(logits, top_k, block_m, has_shared_expert)

    # Fresh-allocation path.
    out = alphamoe_fused_router(
        logits,
        top_k=top_k,
        block_m=block_m,
        has_shared_expert=has_shared_expert,
    )
    _assert_bundle(out, ref)

    # Plan path: buffers preallocated once, refilled per call.
    plan = allocate_alphamoe_route_plan(
        logits, top_k=top_k, block_m=block_m, has_shared_expert=has_shared_expert
    )
    out = alphamoe_fused_router(logits, plan)
    _assert_bundle(out, ref)


@pytest.mark.parametrize("has_shared_expert", [False, True])
def test_alphamoe_fused_router_stable_ties(has_shared_expert):
    """Equal logits must keep the lower expert index first (stable order)."""
    torch.manual_seed(7)
    num_tokens, num_experts, top_k, block_m = 16, 64, 8, 8
    # Small integer alphabet forces many exact ties per row.
    logits = torch.randint(
        -2, 3, (num_tokens, num_experts), dtype=torch.float32, device="cuda"
    )
    ref = _router_reference(logits, top_k, block_m, has_shared_expert)
    out = alphamoe_fused_router(
        logits, top_k=top_k, block_m=block_m, has_shared_expert=has_shared_expert
    )
    _assert_bundle(out, ref)


def test_alphamoe_route_plan_contract():
    num_tokens, num_experts, top_k, block_m = 8, 32, 4, 8
    logits = torch.randn(num_tokens, num_experts, dtype=torch.float32, device="cuda")
    plan = allocate_alphamoe_route_plan(logits, top_k=top_k, block_m=block_m)
    assert isinstance(plan, AlphaMoeRoutePlan)
    max_blocks = _max_blocks(num_tokens, num_experts, top_k, block_m)
    assert plan.topk_weights.shape == (num_tokens, top_k)
    assert plan.topk_ids.shape == (num_tokens, top_k)
    assert plan.sorted_token_ids.shape == (max_blocks * block_m,)
    assert plan.expert_ids.shape == (max_blocks,)
    assert plan.num_tokens_post_padded.shape == (1,)
    assert plan.expert_counts.shape == (num_experts,)
    assert plan.expert_offsets.shape == (num_experts + 1,)
    assert plan.expert_scatter_offsets.shape == (num_experts,)
    # Tuple emulation covers the whole public bundle in canonical order.
    assert len(plan) == 8
    names = (
        "topk_weights",
        "topk_ids",
        "sorted_token_ids",
        "expert_ids",
        "num_tokens_post_padded",
        "expert_counts",
        "expert_offsets",
        "expert_scatter_offsets",
    )
    for tensor, name in zip(plan, names):
        assert tensor is getattr(plan, name)
    # A plan writes into its own persistent buffers across calls.
    out1 = alphamoe_fused_router(logits, plan)
    assert out1[0] is plan.topk_weights
    logits2 = torch.randn_like(logits)
    out2 = alphamoe_fused_router(logits2, plan)
    assert out2[0] is plan.topk_weights
    ref2 = _router_reference(logits2, top_k, block_m, False)
    _assert_bundle(out2, ref2)


def test_alphamoe_fused_router_validation():
    logits = torch.randn(8, 32, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError):
        alphamoe_fused_router(logits)  # missing top_k/block_m without a plan
    with pytest.raises(ValueError):
        alphamoe_fused_router(logits.cpu(), top_k=4, block_m=8)  # CPU input
    with pytest.raises(ValueError):
        alphamoe_fused_router(
            logits.to(torch.float16), top_k=4, block_m=8
        )  # non-fp32 input
    plan = allocate_alphamoe_route_plan(logits, top_k=4, block_m=8)
    mismatched = torch.randn(16, 64, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError):
        alphamoe_fused_router(mismatched, plan)  # geometry mismatch
    with pytest.raises(ValueError):
        alphamoe_fused_router(logits, top_k=0, block_m=8)
    with pytest.raises(BackendSupportedError):
        alphamoe_fused_router(logits, top_k=4, block_m=8, has_shared_expert=False,
                              backend="tensorrt_llm")


def test_alphamoe_fused_router_cuda_graph_replay():
    """Pinned upstream lifecycle: allocate the route plan before warmup and
    capture, capture the routing call, change logits in place at the same
    address, replay, and validate the complete route plan for the new values.
    No replanning and no metadata changes."""
    num_tokens, num_experts, top_k, block_m = 8, 32, 4, 8
    torch.manual_seed(1234)
    logits = torch.randn(num_tokens, num_experts, dtype=torch.float32, device="cuda")
    plan = allocate_alphamoe_route_plan(logits, top_k=top_k, block_m=block_m)

    out = alphamoe_fused_router(logits, plan)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        out = alphamoe_fused_router(logits, plan)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = alphamoe_fused_router(logits, plan)
    stream.synchronize()

    # Runtime data changes without replanning: new values, same addresses.
    logits.copy_(torch.randn_like(logits))
    graph.replay()
    torch.cuda.synchronize()
    ref = _router_reference(logits, top_k, block_m, False)
    _assert_bundle(out, ref)
