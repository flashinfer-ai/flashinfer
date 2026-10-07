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

from __future__ import annotations

import pytest
import torch

from flashinfer.fused_moe.alphamoe_fused_router import (
    AlphaMoERoutePlan,
    _alphamoe_fused_router_impl,
    _fake_alphamoe_fused_router,
    allocate_alphamoe_route_plan,
    alphamoe_fused_router,
)
from flashinfer.jit.cpp_ext import is_cuda_version_at_least
from flashinfer.utils import is_sm100a_supported


def _has_router_gpu() -> bool:
    if not torch.cuda.is_available():
        return False
    device = torch.device("cuda", torch.cuda.current_device())
    capability = torch.cuda.get_device_capability(device)
    return (
        capability in {(10, 0), (10, 3)}
        and is_sm100a_supported(device)
        and is_cuda_version_at_least("12.9" if capability == (10, 3) else "12.8")
    )


requires_router_gpu = pytest.mark.skipif(
    not _has_router_gpu(),
    reason="AlphaMoE fused router requires supported SM100a or SM103a hardware/toolkit",
)


def _reference_topk(
    logits: torch.Tensor, top_k: int, has_shared_expert: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    routed_experts = logits.shape[1] - int(has_shared_expert)
    routed_top_k = top_k - int(has_shared_expert)
    # Random reference inputs have no exact ties. Tie selection/order is outside
    # the public contract and is covered separately below.
    topk_ids = torch.argsort(
        logits[:, :routed_experts], dim=-1, descending=True, stable=True
    )[:, :routed_top_k]
    selected_logits = torch.gather(logits, 1, topk_ids)
    if has_shared_expert:
        shared_ids = torch.full(
            (logits.shape[0], 1),
            logits.shape[1] - 1,
            dtype=torch.int64,
            device=logits.device,
        )
        topk_ids = torch.cat((topk_ids, shared_ids), dim=-1)
        selected_logits = torch.cat((selected_logits, logits[:, -1:]), dim=-1)
    return torch.softmax(selected_logits, dim=-1), topk_ids.to(torch.int32)


def _assert_aligned_plan(
    plan: AlphaMoERoutePlan,
    selected_ids: torch.Tensor,
    *,
    num_experts: int,
    block_m: int,
) -> None:
    num_tokens, top_k = selected_ids.shape
    sentinel = num_tokens * top_k
    selected_ids_cpu = selected_ids.cpu().reshape(-1)
    expected_counts = torch.bincount(
        selected_ids_cpu.to(torch.int64), minlength=num_experts
    ).to(torch.int32)
    padded_counts = ((expected_counts + block_m - 1) // block_m) * block_m
    expected_offsets = torch.empty(num_experts + 1, dtype=torch.int32)
    expected_offsets[0] = 0
    expected_offsets[1:] = torch.cumsum(padded_counts, dim=0)
    extent = int(expected_offsets[-1])

    assert int(plan.num_tokens_post_padded.cpu()[0]) == extent
    torch.testing.assert_close(plan.expert_counts.cpu(), expected_counts)
    torch.testing.assert_close(plan.expert_scatter_offsets.cpu(), expected_counts)
    torch.testing.assert_close(plan.expert_offsets.cpu(), expected_offsets)

    sorted_ids = plan.sorted_token_ids[:extent].cpu()
    expert_ids = plan.expert_ids[: extent // block_m].cpu()
    covered_pairs: list[int] = []
    for expert in range(num_experts):
        start = int(expected_offsets[expert])
        count = int(expected_counts[expert])
        padded_count = int(padded_counts[expert])
        if count == 0:
            continue
        expected_pairs = (
            torch.nonzero(selected_ids_cpu == expert).flatten().to(torch.int32)
        )
        actual_pairs = sorted_ids[start : start + count]
        torch.testing.assert_close(
            torch.sort(actual_pairs).values,
            torch.sort(expected_pairs).values,
            rtol=0,
            atol=0,
        )
        covered_pairs.extend(actual_pairs.tolist())
        assert torch.all(sorted_ids[start + count : start + padded_count] == sentinel)
        first_block = start // block_m
        num_blocks = padded_count // block_m
        assert torch.all(expert_ids[first_block : first_block + num_blocks] == expert)
    assert sorted(covered_pairs) == list(range(sentinel))


def _assert_plan(
    logits: torch.Tensor,
    plan: AlphaMoERoutePlan,
    *,
    top_k: int,
    block_m: int,
    has_shared_expert: bool,
) -> None:
    ref_weights, ref_ids = _reference_topk(logits, top_k, has_shared_expert)
    torch.testing.assert_close(plan.topk_ids, ref_ids, rtol=0, atol=0)
    torch.testing.assert_close(plan.topk_weights, ref_weights, rtol=3e-4, atol=3e-4)

    _assert_aligned_plan(plan, ref_ids, num_experts=logits.shape[1], block_m=block_m)


@requires_router_gpu
@pytest.mark.parametrize(
    "num_tokens,num_experts,top_k,block_m,has_shared_expert",
    [
        (8, 4, 2, 8, False),
        (17, 8, 3, 8, False),
        (8, 257, 9, 8, True),
        (32, 512, 8, 16, False),
        (1, 512, 2, 16, True),
        (9, 16, 1, 1, False),
        (7, 32, 16, 16, False),
    ],
)
def test_alphamoe_fused_router_matches_reference(
    num_tokens, num_experts, top_k, block_m, has_shared_expert
):
    torch.manual_seed(29001)
    logits = torch.randn(num_tokens, num_experts, device="cuda", dtype=torch.float32)
    plan = alphamoe_fused_router(
        logits,
        top_k=top_k,
        block_m=block_m,
        has_shared_expert=has_shared_expert,
    )
    _assert_plan(
        logits,
        plan,
        top_k=top_k,
        block_m=block_m,
        has_shared_expert=has_shared_expert,
    )


@requires_router_gpu
@pytest.mark.parametrize("has_shared_expert", [False, True])
def test_alphamoe_fused_router_ties_form_valid_plan(has_shared_expert):
    logits = torch.zeros(3, 512, device="cuda", dtype=torch.float32)
    plan = alphamoe_fused_router(
        logits,
        top_k=8,
        block_m=16,
        has_shared_expert=has_shared_expert,
    )
    routed_experts = 512 - int(has_shared_expert)
    routed_top_k = 8 - int(has_shared_expert)
    routed_ids = plan.topk_ids[:, :routed_top_k]
    assert torch.all((routed_ids >= 0) & (routed_ids < routed_experts))
    assert torch.all(torch.sort(routed_ids, dim=1).values.diff(dim=1) > 0)
    if has_shared_expert:
        assert torch.all(plan.topk_ids[:, -1] == 511)
    selected_logits = torch.gather(logits, 1, plan.topk_ids.to(torch.int64))
    torch.testing.assert_close(
        plan.topk_weights,
        torch.softmax(selected_logits, dim=-1),
        rtol=3e-4,
        atol=3e-4,
    )
    _assert_aligned_plan(
        plan,
        plan.topk_ids,
        num_experts=512,
        block_m=16,
    )


@requires_router_gpu
def test_alphamoe_fused_router_large_persistent_grid_and_hot_expert():
    device = torch.cuda.current_device()
    num_tokens = 2 * torch.cuda.get_device_properties(device).multi_processor_count + 3
    logits = torch.full((num_tokens, 32), -20.0, device="cuda", dtype=torch.float32)
    logits[:, :4] = torch.tensor([10.0, 9.0, 8.0, 7.0], device="cuda")
    plan = alphamoe_fused_router(logits, top_k=4, block_m=8)
    _assert_plan(logits, plan, top_k=4, block_m=8, has_shared_expert=False)


@requires_router_gpu
def test_alphamoe_fused_router_reuses_poisoned_oversized_plan():
    torch.manual_seed(17)
    logits = torch.randn(11, 64, device="cuda", dtype=torch.float32)
    base = allocate_alphamoe_route_plan(logits, top_k=5, block_m=8)
    sorted_capacity = base.sorted_token_ids.numel()
    block_capacity = base.expert_ids.numel()
    plan = base._replace(
        sorted_token_ids=torch.full(
            (sorted_capacity + 37,), -1234567, device="cuda", dtype=torch.int32
        ),
        expert_ids=torch.full(
            (block_capacity + 11,), -7654321, device="cuda", dtype=torch.int32
        ),
    )
    pointers = tuple(t.data_ptr() for t in plan)
    returned = alphamoe_fused_router(logits, top_k=5, block_m=8, plan=plan)
    assert returned is plan
    assert tuple(t.data_ptr() for t in returned) == pointers
    _assert_plan(logits, plan, top_k=5, block_m=8, has_shared_expert=False)
    assert torch.all(plan.sorted_token_ids[sorted_capacity:] == -1234567)
    assert torch.all(plan.expert_ids[block_capacity:] == -7654321)


@requires_router_gpu
def test_alphamoe_fused_router_custom_op_returns_none():
    logits = torch.randn(2, 8, device="cuda", dtype=torch.float32)
    plan = allocate_alphamoe_route_plan(logits, top_k=2, block_m=8)
    result = _alphamoe_fused_router_impl(
        logits,
        *plan,
        2,
        8,
        False,
    )
    assert result is None
    _assert_plan(logits, plan, top_k=2, block_m=8, has_shared_expert=False)


def test_alphamoe_fused_router_custom_op_registration_is_module_level():
    assert "<locals>" not in _alphamoe_fused_router_impl.__qualname__
    assert "<locals>" not in _fake_alphamoe_fused_router.__qualname__


@requires_router_gpu
def test_alphamoe_fused_router_torch_compile():
    """Exercise Dynamo under this repo's identity custom-op decorators.

    FlashInfer currently disables the underlying torch.library registrations in
    ``utils.py``. This still guards the supported graph-break path; the
    module-level impl/fake layout is ready for fullgraph schema/meta tracing if
    those global decorators are re-enabled.
    """

    logits = torch.randn(8, 32, device="cuda", dtype=torch.float32)
    plan = allocate_alphamoe_route_plan(logits, top_k=4, block_m=8)
    # Keep extension compilation and loading outside Dynamo tracing.
    alphamoe_fused_router(logits, top_k=4, block_m=8, plan=plan)

    def run(input_logits: torch.Tensor):
        result = alphamoe_fused_router(input_logits, top_k=4, block_m=8, plan=plan)
        return result.topk_weights, result.topk_ids

    compiled = torch.compile(run, backend="eager", fullgraph=False)
    logits.normal_()
    weights, ids = compiled(logits)
    assert weights.data_ptr() == plan.topk_weights.data_ptr()
    assert ids.data_ptr() == plan.topk_ids.data_ptr()
    _assert_plan(logits, plan, top_k=4, block_m=8, has_shared_expert=False)


@requires_router_gpu
def test_alphamoe_fused_router_uses_current_stream():
    logits = torch.randn(13, 32, device="cuda", dtype=torch.float32)
    plan = allocate_alphamoe_route_plan(logits, top_k=4, block_m=8)
    # Warm JIT/module setup outside the non-default stream.
    alphamoe_fused_router(logits, top_k=4, block_m=8, plan=plan)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        logits.normal_()
        alphamoe_fused_router(logits, top_k=4, block_m=8, plan=plan)
    torch.cuda.current_stream().wait_stream(stream)
    _assert_plan(logits, plan, top_k=4, block_m=8, has_shared_expert=False)


@requires_router_gpu
def test_alphamoe_fused_router_cuda_graph_replay():
    logits = torch.randn(8, 32, device="cuda", dtype=torch.float32)
    plan = allocate_alphamoe_route_plan(logits, top_k=4, block_m=8)
    alphamoe_fused_router(logits, top_k=4, block_m=8, plan=plan)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        alphamoe_fused_router(logits, top_k=4, block_m=8, plan=plan, skip_check=True)
    logits.copy_(torch.randn_like(logits))
    graph.replay()
    torch.cuda.synchronize()
    _assert_plan(logits, plan, top_k=4, block_m=8, has_shared_expert=False)


@requires_router_gpu
@pytest.mark.parametrize(
    "shape,top_k,block_m,shared,error",
    [
        ((0, 8), 2, 8, False, "num_tokens"),
        ((2, 513), 2, 8, False, "num_experts"),
        ((2, 8), 0, 8, False, "top_k"),
        ((2, 8), 9, 8, False, "top_k"),
        ((2, 8), 2, 0, False, "block_m"),
        ((2, 8), 2, 17, False, "block_m"),
        ((2, 8), 1, 8, True, "shared expert"),
    ],
)
def test_alphamoe_fused_router_rejects_invalid_problem(
    shape, top_k, block_m, shared, error
):
    logits = torch.empty(shape, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match=error):
        alphamoe_fused_router(
            logits,
            top_k=top_k,
            block_m=block_m,
            has_shared_expert=shared,
        )


@requires_router_gpu
def test_alphamoe_fused_router_rejects_bad_tensor_and_plan():
    logits = torch.randn(4, 8, device="cuda", dtype=torch.float32)
    with pytest.raises(TypeError, match="float32"):
        allocate_alphamoe_route_plan(logits.to(torch.float16), top_k=2)
    with pytest.raises(ValueError, match="contiguous"):
        allocate_alphamoe_route_plan(logits[:, ::2], top_k=2)

    plan = allocate_alphamoe_route_plan(logits, top_k=2, block_m=8)
    with pytest.raises(ValueError, match="capacity"):
        alphamoe_fused_router(
            logits,
            top_k=2,
            block_m=8,
            plan=plan._replace(sorted_token_ids=plan.sorted_token_ids[:-1]),
        )

    alias_plan = plan._replace(
        topk_ids=plan.expert_counts[: logits.shape[0] * 2].view(logits.shape[0], 2)
    )
    with pytest.raises(RuntimeError, match="overlap"):
        alphamoe_fused_router(logits, top_k=2, block_m=8, plan=alias_plan)
