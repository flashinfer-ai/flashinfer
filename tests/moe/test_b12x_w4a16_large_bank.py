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

The kernel's own offsets into a bank are a second, independent overflow: with
``Int32`` arithmetic, ``expert_idx * words_per_expert`` wraps once a packed bank
crosses 2**31 int32 words (8 GiB) and ``cp.async`` reads an invalid address.
Those offsets are ``Int64`` now; the second test routes to the last experts of
such a bank.
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


# Packed w13 bank just past 2**31 int32 words (8 GiB) plus its w2 bank (4 GiB),
# E4M3 scales (1.5 GiB), a two-expert copy of the tail and activations.
_REQUIRED_VRAM_PACKED_TAIL = 16 * 1024**3


@sm120_required
@pytest.mark.skipif(
    _free_vram_bytes() < _REQUIRED_VRAM_PACKED_TAIL,
    reason="needs ~16 GiB free VRAM for a packed expert weight bank past 8 GiB",
)
def test_w4a16_moe_packed_bank_expert_offset_over_int32_max():
    """The last experts of a packed bank past 2**31 int32 words read correctly.

    Routes every token to the two last experts of a 129-expert packed bank and
    compares against the same two experts copied into a two-expert bank, where
    no offset can wrap. Before the Int64 offsets, expert 128's block started at
    word 2**31 and the wrapped Int32 offset made cp.async fault or read other
    memory.
    """
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
    # Packed layout: E * (k // 16) * (n // 16 * 32) int32 words per bank.
    # 129 * 8192 * 16384 // 8 = 2,164,260,864 words, so the last expert's block
    # begins at word 128 * 16,777,216 = 2**31.
    num_experts, hidden_size, intermediate_size = 129, 8192, 8192
    top_k = 2
    fc1_cols = 2 * intermediate_size
    w13_words_per_expert = hidden_size * fc1_cols // 8
    w2_words_per_expert = intermediate_size * hidden_size // 8
    assert (num_experts - 1) * w13_words_per_expert == 2**31
    tail_first = num_experts - top_k

    def random_words(words):
        return torch.randint(
            0, 256, (words * 4,), dtype=torch.uint8, device=device
        ).view(torch.int32)

    w13 = random_words(num_experts * w13_words_per_expert)
    w2 = random_words(num_experts * w2_words_per_expert)
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
    # 0x08 is E4M3 2**-6: small dequantized weights keep the bf16 outputs of
    # the two differently scheduled runs comparable at a loose tolerance.
    w13_scale = torch.full((n13,), 0x08080808, dtype=torch.int32, device=device)
    w2_scale = torch.full((n2,), 0x08080808, dtype=torch.int32, device=device)

    def packed(w13_bank, w13_scale_bank, w2_bank, w2_scale_bank, experts):
        return W4A16PackedWeights(
            w13=w13_bank,
            w13_scale=w13_scale_bank,
            w13_global_scale=torch.ones(experts, dtype=torch.float32, device=device),
            w2=w2_bank,
            w2_scale=w2_scale_bank,
            w2_global_scale=torch.ones(experts, dtype=torch.float32, device=device),
            workspace=_make_workspace(device),
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=experts,
            is_gated=True,
            params_dtype=torch.bfloat16,
            source_format="modelopt",
            w13_layout="w13",
            weight_layout="packed",
            scale_format="e4m3_k16",
        )

    large = packed(w13, w13_scale, w2, w2_scale, num_experts)
    # Weight and scale banks are expert-major, so the tail experts' blocks form
    # a valid two-expert bank on their own.
    small = packed(
        w13[tail_first * w13_words_per_expert :].clone(),
        w13_scale[tail_first * (n13 // num_experts) :].clone(),
        w2[tail_first * w2_words_per_expert :].clone(),
        w2_scale[tail_first * (n2 // num_experts) :].clone(),
        top_k,
    )

    props = torch.cuda.get_device_properties(device)
    sms = int(props.multi_processor_count)

    def run(prepared, expert_ids, a):
        m = int(a.shape[0])
        plan = plan_w4a16_buffers(
            prepared,
            m=m,
            topk=top_k,
            route_num_experts=int(prepared.num_experts),
            sms=sms,
        )
        topk_ids = torch.tensor([expert_ids], dtype=torch.int32, device=device)
        topk_weights = torch.tensor([[0.75, 0.25]], dtype=torch.float32, device=device)
        output = torch.zeros(m, hidden_size, dtype=torch.bfloat16, device=device)
        run_w4a16_moe(
            a,
            prepared,
            topk_weights.repeat(m, 1).contiguous(),
            topk_ids.repeat(m, 1).contiguous(),
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
        return output

    # Decode-band and general-band token counts.
    for m in (4, 16):
        a = torch.randn(m, hidden_size, dtype=torch.bfloat16, device=device)
        expected = run(small, [0, 1], a)
        actual = run(large, [tail_first, tail_first + 1], a)
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=5e-2)
        # A different expert pair must not match, or the check above is vacuous.
        other = run(large, [0, 1], a)
        assert not torch.allclose(other, expected, rtol=2e-2, atol=5e-2)
