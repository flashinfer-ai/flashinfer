"""Regression test: SM12x W4A16 MoE with a flat expert weight bank > 2**31
elements must not overflow during host-side argument marshalling.

The fused-MoE launch flattens each expert weight bank to a rank-1 tensor. In
the "modelopt" weight layout that flat view is uint8, so a bank larger than
2 GiB exceeds int32 range; the implicit TensorAdapter conversion marks the
layout dynamic, and dynamic sizes are packed 32-bit by the DSL runtime,
raising ``OverflowError: Value overflow: ... exceeds range of l`` before the
kernel ever launches. The launch now wraps the banks in a static-layout
``from_dlpack`` (matching the static-shape fakes the kernel is compiled
against), which keeps sizes 64-bit.
"""

import pytest
import torch


def _is_sm12x_supported():
    if not torch.cuda.is_available():
        return False
    from flashinfer.utils import is_sm120a_supported, is_sm121a_supported

    device = torch.device("cuda")
    return is_sm120a_supported(device) or is_sm121a_supported(device)


def _free_vram_bytes():
    if not torch.cuda.is_available():
        return 0
    free, _total = torch.cuda.mem_get_info()
    return free


sm120_required = pytest.mark.skipif(
    not _is_sm12x_supported(),
    reason="W4A16 fused MoE requires SM120/SM121",
)

# w13 bank 2.5 GiB + w2 bank 1.25 GiB + scales/activations head-room.
_REQUIRED_VRAM = 6 * 1024**3


@sm120_required
@pytest.mark.skipif(
    _free_vram_bytes() < _REQUIRED_VRAM,
    reason="needs ~6 GiB free VRAM for a >2 GiB expert weight bank",
)
def test_w4a16_moe_flat_bank_over_int32_max():
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_host import (
        plan_w4a16_buffers,
    )
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_kernel import (
        _scale_fake_int32_elements,
        run_w4a16_moe,
    )
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_prepare import (
        W4A16PackedWeights,
        _make_workspace,
    )

    device = torch.device("cuda")
    torch.manual_seed(0)
    # E * (2 * n) * (k // 2) = 64 * 8192 * 5120 = 2,684,354,560 uint8 elements:
    # the smallest realistic bank past int32 max (matches the report in the
    # linked issue, Qwen3.6-35B-A3B-NVFP4 on GB10).
    num_experts, hidden_size, intermediate_size = 64, 10240, 4096
    m, top_k = 4, 2
    fc1_cols = 2 * intermediate_size
    bank_elements = num_experts * fc1_cols * (hidden_size // 2)
    assert bank_elements > 2**31 - 1

    w13 = torch.randint(
        0,
        256,
        (num_experts, fc1_cols, hidden_size // 2),
        dtype=torch.uint8,
        device=device,
    )
    w2 = torch.randint(
        0,
        256,
        (num_experts, hidden_size, intermediate_size // 2),
        dtype=torch.uint8,
        device=device,
    )
    n13 = _scale_fake_int32_elements(
        num_experts=num_experts,
        size_k=hidden_size,
        size_n=fc1_cols,
        scale_format="e4m3_k16",
    )
    n2 = _scale_fake_int32_elements(
        num_experts=num_experts,
        size_k=intermediate_size,
        size_n=hidden_size,
        scale_format="e4m3_k16",
    )
    prepared = W4A16PackedWeights(
        w13=w13,
        w13_scale=torch.full((n13,), 0x3C3C3C3C, dtype=torch.int32, device=device),
        w13_global_scale=torch.ones(num_experts, dtype=torch.float32, device=device),
        w2=w2,
        w2_scale=torch.full((n2,), 0x3C3C3C3C, dtype=torch.int32, device=device),
        w2_global_scale=torch.ones(num_experts, dtype=torch.float32, device=device),
        workspace=_make_workspace(device),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        is_gated=True,
        params_dtype=torch.bfloat16,
        source_format="modelopt",
        w13_layout="w13",
        weight_layout="modelopt",
        scale_format="e4m3_k16",
    )

    a = torch.randn(m, hidden_size, dtype=torch.bfloat16, device=device)
    logits = torch.randn(m, num_experts, dtype=torch.float32, device=device)
    topk_weights, topk_ids = torch.topk(torch.softmax(logits, -1), top_k)
    props = torch.cuda.get_device_properties(device)
    plan = plan_w4a16_buffers(
        prepared,
        m=m,
        topk=top_k,
        route_num_experts=num_experts,
        sms=int(props.multi_processor_count),
    )
    output = torch.zeros(m, hidden_size, dtype=torch.bfloat16, device=device)

    # Overflowed at host-side argument marshalling before the fix.
    run_w4a16_moe(
        a,
        prepared,
        topk_weights.float().contiguous(),
        topk_ids.to(torch.int32).contiguous(),
        activation="silu",
        intermediate_cache13=torch.empty(
            plan.intermediate_cache13_elements, dtype=torch.bfloat16, device=device
        ),
        intermediate_cache2=torch.empty(
            plan.intermediate_cache2_elements, dtype=torch.bfloat16, device=device
        ),
        output=output,
    )
    torch.cuda.synchronize()
    assert torch.isfinite(output).all()
