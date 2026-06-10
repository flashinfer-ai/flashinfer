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

from flashinfer.fused_moe import hash_topk


def _ref_hash_topk(
    router_logits: torch.Tensor,
    input_ids: torch.Tensor,
    tid2eid: torch.Tensor,
    num_fused_shared_experts: int,
    routed_scaling_factor: float,
):
    """Torch reference matching SGLang hash_topk._forward_torch (sqrtsoftplus)."""
    num_tokens, num_routed_experts = router_logits.shape
    topk = tid2eid.shape[1]
    topk_fused = topk + num_fused_shared_experts

    scores = torch.sqrt(torch.nn.functional.softplus(router_logits))  # [N, E]
    expert_ids = tid2eid[input_ids].to(torch.int64)  # [N, topk]
    gathered = torch.gather(scores, 1, expert_ids)  # [N, topk]
    routed_sum = gathered.sum(dim=-1, keepdim=True)  # [N, 1]

    weights = torch.empty(
        (num_tokens, topk_fused), dtype=torch.float32, device=router_logits.device
    )
    ids = torch.empty(
        (num_tokens, topk_fused), dtype=torch.int32, device=router_logits.device
    )
    weights[:, :topk] = gathered / routed_sum
    ids[:, :topk] = expert_ids.to(torch.int32)
    for s in range(num_fused_shared_experts):
        ids[:, topk + s] = num_routed_experts + s
        weights[:, topk + s] = 1.0 / routed_scaling_factor
    return weights, ids


@pytest.mark.parametrize("num_tokens", [1, 7, 64, 512, 4096])
@pytest.mark.parametrize("num_experts", [256, 384])
@pytest.mark.parametrize("topk", [8])
@pytest.mark.parametrize("num_fused_shared_experts", [0, 1])
@pytest.mark.parametrize("launch_with_pdl", [False, True])
def test_hash_topk(
    num_tokens, num_experts, topk, num_fused_shared_experts, launch_with_pdl
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = torch.device("cuda")
    torch.manual_seed(0)

    vocab = 1024
    router_logits = torch.randn(
        num_tokens, num_experts, dtype=torch.float32, device=device
    )
    input_ids = torch.randint(0, vocab, (num_tokens,), dtype=torch.int64, device=device)
    # Each token-row of the table holds `topk` distinct routed-expert ids.
    tid2eid = torch.empty((vocab, topk), dtype=torch.int32, device=device)
    for v in range(vocab):
        perm = torch.randperm(num_experts, device=device)[:topk]
        tid2eid[v] = perm.to(torch.int32)

    routed_scaling_factor = 2.5

    weights, ids = hash_topk(
        router_logits,
        input_ids,
        tid2eid,
        num_fused_shared_experts=num_fused_shared_experts,
        routed_scaling_factor=routed_scaling_factor,
        launch_with_pdl=launch_with_pdl,
    )

    ref_weights, ref_ids = _ref_hash_topk(
        router_logits,
        input_ids,
        tid2eid,
        num_fused_shared_experts,
        routed_scaling_factor,
    )

    topk_fused = topk + num_fused_shared_experts
    assert weights.shape == (num_tokens, topk_fused)
    assert ids.shape == (num_tokens, topk_fused)
    assert weights.dtype == torch.float32
    assert ids.dtype == torch.int32

    torch.testing.assert_close(weights, ref_weights, rtol=1e-3, atol=1e-3)
    assert torch.equal(ids, ref_ids)


def _make_inputs(num_tokens, num_experts, topk, device, vocab=256):
    router_logits = torch.randn(
        num_tokens, num_experts, dtype=torch.float32, device=device
    )
    input_ids = torch.randint(
        0, vocab, (max(num_tokens, 1),), dtype=torch.int64, device=device
    )[:num_tokens]
    tid2eid = torch.empty((vocab, topk), dtype=torch.int32, device=device)
    for v in range(vocab):
        tid2eid[v] = torch.randperm(num_experts, device=device)[:topk].to(torch.int32)
    return router_logits, input_ids, tid2eid


@pytest.mark.parametrize("topk", [1, 31])
def test_hash_topk_topk_fused_boundary(topk):
    """topk + shared == 32 (warp size) must be accepted."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = torch.device("cuda")
    torch.manual_seed(2)
    num_experts = 384
    router_logits, input_ids, tid2eid = _make_inputs(64, num_experts, topk, device)
    weights, ids = hash_topk(
        router_logits, input_ids, tid2eid, num_fused_shared_experts=1
    )
    assert weights.shape == (64, topk + 1)
    assert ids.shape == (64, topk + 1)


def test_hash_topk_zero_tokens():
    """num_tokens == 0 must return empty outputs without launching a 0-grid."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = torch.device("cuda")
    router_logits, input_ids, tid2eid = _make_inputs(0, 256, 8, device)
    weights, ids = hash_topk(
        router_logits, input_ids, tid2eid, num_fused_shared_experts=1
    )
    assert weights.shape == (0, 9)
    assert ids.shape == (0, 9)


def test_hash_topk_invalid_args():
    """Out-of-contract arguments must raise rather than silently misbehave."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = torch.device("cuda")
    router_logits, input_ids, tid2eid = _make_inputs(16, 256, 8, device)

    # topk_fused (= topk + shared) > 32
    big_tid2eid = torch.zeros((256, 32), dtype=torch.int32, device=device)
    with pytest.raises(ValueError):
        hash_topk(router_logits, input_ids, big_tid2eid, num_fused_shared_experts=1)

    # num_fused_shared_experts not in {0, 1}
    with pytest.raises(ValueError):
        hash_topk(router_logits, input_ids, tid2eid, num_fused_shared_experts=2)

    # routed_scaling_factor <= 0 with a fused shared expert
    with pytest.raises(ValueError):
        hash_topk(
            router_logits,
            input_ids,
            tid2eid,
            num_fused_shared_experts=1,
            routed_scaling_factor=0.0,
        )

    # non-contiguous router_logits
    non_contig = torch.randn(16, 512, dtype=torch.float32, device=device)[:, ::2]
    with pytest.raises(ValueError):
        hash_topk(non_contig, input_ids, tid2eid)

    # wrong dtype
    with pytest.raises(ValueError):
        hash_topk(router_logits.to(torch.bfloat16), input_ids, tid2eid)
