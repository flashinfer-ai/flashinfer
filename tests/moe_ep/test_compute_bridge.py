"""Unit tests for the EP→compute layout bridge.

Verifies the token-major translation of the expert-major dispatch output:
flatten ``[E_local, cap, hidden] → [E_local*cap, hidden]``, synthesized
``selected_experts = row // cap + local_expert_offset``, ``final_scales == 1``,
``top_k == 1``, and the reshape back to the 3D combine layout.

The bf16 path is host-checkable (no quant kernel); the NVFP4 path needs an
SM100+ device for ``fp4_quantize`` and is skipped otherwise.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from flashinfer.moe_ep._compute_bridge import (  # noqa: E402
    build_activation_pack,
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
def test_reshape_for_combine_roundtrips():
    num_local_experts, cap, hidden = 4, 8, 128
    et = _dispatch_tensor(num_local_experts, cap, hidden, "cuda")
    flat = et.reshape(num_local_experts * cap, hidden)

    back = reshape_for_combine(flat, num_local_experts, cap)
    assert back.shape == (num_local_experts, cap, hidden)
    assert torch.equal(back, et)


def test_build_activation_pack_rejects_2d():
    pytest.importorskip("torch")
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
    # Packed FP4: last dim halves; scale factor present.
    assert pack.hidden_states_q.shape[0] == m
    assert pack.hidden_states_q.shape[1] == hidden // 2
    assert pack.hidden_states_scale.numel() > 0
    assert pack.selected_experts.shape == (m, 1)
