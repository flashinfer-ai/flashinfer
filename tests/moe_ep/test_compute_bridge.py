"""Unit tests for the EP→compute layout bridge.

Verifies the token-major translation of both LL dispatch layouts:

* EXPERT_MAJOR: flatten ``[E_local, cap, hidden] → [E_local*cap, hidden]``,
  synthesized ``selected_experts = row // cap + local_expert_offset``.
* RANK_MAJOR: flatten ``[world, per_rank, hidden] → [world*per_rank, hidden]``,
  driven by the received per-token ``topk_idx`` / ``topk_weights`` at the real
  model ``top_k`` with non-local picks masked to weight 0.

EXPERT_MAJOR synthesizes ``final_scales == 1`` / ``top_k == 1``; both reshape back
to the 3D combine layout.

The bf16 path is host-checkable (no quant kernel); the NVFP4 path needs an
SM100+ device for ``fp4_quantize`` and is skipped otherwise.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from flashinfer.moe_ep.backends.split.kernel.fused_moe.bridge import (  # noqa: E402
    build_activation_pack,
    build_activation_pack_rank_major,
    reshape_for_combine,
)


def _dispatch_tensor(num_local_experts, cap, hidden, device):
    return torch.randn(
        num_local_experts, cap, hidden, dtype=torch.bfloat16, device=device
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_bf16_pack_shapes_and_routing():
    num_local_experts, cap, hidden = 4, 8, 128
    offset = 12  # rank 3 of an 8-expert/rank shard, say
    et = _dispatch_tensor(num_local_experts, cap, hidden, "cuda")

    pack = build_activation_pack(et, local_expert_offset=offset, is_nvfp4=False)

    m = num_local_experts * cap
    assert pack.hidden_states_q.shape == (m, hidden)
    assert pack.hidden_states_q.dtype == torch.bfloat16  # raw passthrough
    assert pack.selected_experts.shape == (m, 1)  # top_k == 1
    assert pack.final_scales.shape == (m, 1)
    assert torch.all(pack.final_scales == 1.0)  # combine owns the reweight

    # Row r belongs to local expert r // cap, shifted by the global offset.
    expected = (torch.arange(m, device="cuda") // cap).to(torch.int32) + offset
    assert torch.equal(pack.selected_experts.squeeze(-1), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_rank_major_pack_faithful_routing_and_masking():
    # RANK_MAJOR recv is [world, max_tokens_per_rank, hidden]; compute is driven
    # by the received per-token routing, masked to this rank's local experts.
    world, per_rank, hidden = 8, 16, 128
    top_k = 8
    num_local_experts = 4
    offset = 12  # local window = [12, 16)
    m = world * per_rank
    et = _dispatch_tensor(world, per_rank, hidden, "cuda")

    g = torch.Generator(device="cuda").manual_seed(0)
    # RANK_MAJOR dispatch returns LOCAL expert indices (0-based within this rank's
    # experts), with -1 marking a pick routed to a non-local expert (owned by
    # another rank). Generate that convention: ids in [-1, num_local_experts).
    recv_idx = torch.randint(
        -1, num_local_experts, (m, top_k), device="cuda", dtype=torch.int64, generator=g
    )
    recv_w = torch.rand(m, top_k, device="cuda", generator=g)

    pack = build_activation_pack_rank_major(
        et,
        recv_idx,
        recv_w,
        num_local_experts=num_local_experts,
        local_expert_offset=offset,
        is_nvfp4=False,
    )

    assert pack.hidden_states_q.shape == (m, hidden)
    assert pack.hidden_states_q.dtype == torch.bfloat16  # raw passthrough
    assert pack.selected_experts.shape == (m, top_k)  # real model top_k
    assert pack.final_scales.shape == (m, top_k)

    is_local = recv_idx >= 0
    # Local picks: local id -> global (id + offset), real weight kept. Non-local
    # picks (-1) are pinned to the first local expert (offset) with weight 0
    # (dropped by the weighted finalize).
    assert torch.equal(
        pack.selected_experts[is_local], (recv_idx[is_local] + offset).to(torch.int32)
    )
    assert torch.all(pack.selected_experts[~is_local] == offset)
    assert torch.allclose(pack.final_scales[is_local], recv_w[is_local])
    assert torch.all(pack.final_scales[~is_local] == 0.0)
    # Every synthesized expert id lands inside this rank's local-expert range.
    assert pack.selected_experts.min().item() >= offset
    assert pack.selected_experts.max().item() < offset + num_local_experts


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_rank_major_masks_out_of_range_local_expert_ids():
    world, per_rank, hidden = 4, 8, 64
    top_k = 4
    num_local_experts = 4
    offset = 8
    m = world * per_rank
    et = _dispatch_tensor(world, per_rank, hidden, "cuda")

    recv_idx = torch.full(
        (m, top_k), num_local_experts, dtype=torch.int64, device="cuda"
    )
    recv_w = torch.ones(m, top_k, device="cuda")

    pack = build_activation_pack_rank_major(
        et,
        recv_idx,
        recv_w,
        num_local_experts=num_local_experts,
        local_expert_offset=offset,
        is_nvfp4=False,
    )

    assert torch.all(pack.selected_experts == offset)
    assert torch.all(pack.final_scales == 0.0)


def test_build_activation_pack_rank_major_rejects_2d():
    bad = torch.zeros(8, 128, dtype=torch.bfloat16)
    idx = torch.zeros(8, 8, dtype=torch.int64)
    w = torch.zeros(8, 8, dtype=torch.float32)
    with pytest.raises(ValueError):
        build_activation_pack_rank_major(
            bad, idx, w, num_local_experts=4, is_nvfp4=False
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_reshape_for_combine_roundtrips():
    num_local_experts, cap, hidden = 4, 8, 128
    et = _dispatch_tensor(num_local_experts, cap, hidden, "cuda")
    flat = et.reshape(num_local_experts * cap, hidden)

    back = reshape_for_combine(flat, num_local_experts, cap)
    assert back.shape == (num_local_experts, cap, hidden)
    assert torch.equal(back, et)


def test_build_activation_pack_rejects_2d():
    bad = torch.zeros(8, 128, dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        build_activation_pack(bad, is_nvfp4=False)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="NVFP4 fp4_quantize needs SM100+",
)
def test_nvfp4_pack_quantizes():
    num_local_experts, cap, hidden = 4, 8, 128
    et = _dispatch_tensor(num_local_experts, cap, hidden, "cuda")

    pack = build_activation_pack(et, local_expert_offset=0, is_nvfp4=True)

    m = num_local_experts * cap
    assert pack.hidden_states_q.shape[0] == m
    assert pack.hidden_states_q.shape[1] == hidden // 2
    assert pack.hidden_states_scale.numel() > 0
    assert pack.selected_experts.shape == (m, 1)
