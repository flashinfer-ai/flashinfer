"""Check MXFP8 weight preparation and standard inference against Torch.

The public weight transform must use the sources kernel's 16-row gate/up
interleave. The forward check compares with independently reduced expert terms.

Run on one Blackwell GPU from the FlashInfer repo root (no torchrun required)::

    cd /path/to/flashinfer
    export PYTHONPATH="${PWD}:${PYTHONPATH}"
    MEGA_NO_DIST=1 CUDA_VISIBLE_DEVICES=0 pytest \\
        tests/moe_ep/test_mxfp8_cutedsl_preprocess_vs_reference.py -v \\
        -m arch_blackwell --confcutdir=tests/moe_ep
"""

from __future__ import annotations

import pytest

# This test verifies MXFP8 preprocess + kernel output only through the
# cutedsl_megamoe shim public API (including the torch reference it re-exports);
# it never imports the src/ kernel packages directly, so a new src/ drop can't
# silently break it.
pytest.importorskip("flashinfer.moe_ep.shims.sm100.cutedsl_megamoe.sources")


def _require_cuda():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")


def _single_rank_problem():
    import torch

    hidden = 2048
    intermediate = 1024
    num_tokens = 32
    max_tokens = 64
    num_experts = 4
    topk = 4
    num_local_experts = num_experts
    gate_up_clamp = 10.0
    kind = "mxfp8_e4m3"

    g = torch.Generator(device="cuda").manual_seed(7)
    hidden_states = torch.randn(
        num_tokens, hidden, dtype=torch.bfloat16, device="cuda", generator=g
    )
    scores = torch.randn(
        num_tokens, num_experts, dtype=torch.float32, device="cuda", generator=g
    )
    topk_weights, topk_ids = torch.topk(
        scores, topk, dim=-1, largest=True, sorted=False
    )

    g = torch.Generator(device="cuda").manual_seed(13)
    w13 = torch.randn(
        num_local_experts,
        2 * intermediate,
        hidden,
        dtype=torch.bfloat16,
        device="cuda",
        generator=g,
    )
    w2 = torch.randn(
        num_local_experts,
        hidden,
        intermediate,
        dtype=torch.bfloat16,
        device="cuda",
        generator=g,
    )

    return dict(
        hidden=hidden,
        intermediate=intermediate,
        num_tokens=num_tokens,
        max_tokens=max_tokens,
        num_experts=num_experts,
        topk=topk,
        gate_up_clamp=gate_up_clamp,
        kind=kind,
        hidden_states=hidden_states,
        topk_weights=topk_weights.to(torch.float32),
        topk_ids=topk_ids.to(torch.int64),
        w13=w13,
        w2=w2,
    )


def _plain_mxfp8_from_bf16(problem: dict):
    """bf16 weights → kernel fp8 + plain E8M0 SF (``mega_reference`` layout)."""
    import torch

    from flashinfer.moe_ep import MoEWeightPack
    from flashinfer.moe_ep.backends.mega.kernel.sm100.mxfp8_mxfp8_bf16_cutedsl.weights import (
        _quantize_mxfp8_weight_k_major,
    )

    intermediate_size = problem["intermediate"]
    kind = problem["kind"]
    pack = MoEWeightPack(w13=problem["w13"], w2=problem["w2"])
    num_experts = pack.w13.shape[0]

    gate, up = pack.w13.split(intermediate_size, dim=1)
    fc1_fp32 = (
        torch.cat(
            [
                torch.cat(
                    (gate[:, start : start + 16], up[:, start : start + 16]), dim=1
                )
                for start in range(0, intermediate_size, 16)
            ],
            dim=1,
        )
        .transpose(1, 2)
        .contiguous()
    )
    fc1_weights = []
    fc1_plain_sf = []
    fc2_weights = []
    fc2_plain_sf = []
    for expert in range(num_experts):
        fc1_q, fc1_sf = _quantize_mxfp8_weight_k_major(
            fc1_fp32[expert].transpose(0, 1),
            kind=kind,
        )
        fc1_weights.append(fc1_q.transpose(0, 1))
        fc1_plain_sf.append(fc1_sf)

        fc2_hw = pack.w2[expert]
        fc2_q, fc2_sf = _quantize_mxfp8_weight_k_major(
            fc2_hw,
            kind=kind,
        )
        fc2_weights.append(fc2_q.transpose(0, 1))
        fc2_plain_sf.append(fc2_sf)

    return (
        torch.stack(fc1_weights, dim=0),
        torch.stack(fc1_plain_sf, dim=0),
        torch.stack(fc2_weights, dim=0),
        torch.stack(fc2_plain_sf, dim=0),
    )


# Use the same tolerance as the SM120 oracle.
from .mega_oracle_compare import (  # noqa: E402
    _assert_mega_oracle_term_band_close,
)


@pytest.mark.arch_blackwell
def test_mxfp8_preprocess_fp8_weights_match_plain_quant():
    """``preprocess_mega_weights`` fp8 tensors match an independent plain quant."""
    _require_cuda()

    import torch

    from flashinfer.moe_ep import MoEWeightPack
    from flashinfer.moe_ep.backends.mega.kernel.sm100.mxfp8_mxfp8_bf16_cutedsl.weights import (
        preprocess_mega_weights,
    )

    problem = _single_rank_problem()
    pack = MoEWeightPack(w13=problem["w13"], w2=problem["w2"])

    transformed_l1, transformed_l2 = preprocess_mega_weights(
        pack,
        intermediate_size=problem["intermediate"],
        hidden_size=problem["hidden"],
        kind=problem["kind"],
        gate_up_clamp=problem["gate_up_clamp"],
    )
    fc1_kernel, _fc1_swz = transformed_l1
    fc2_kernel, _fc2_swz = transformed_l2

    fc1_plain, _fc1_sf, fc2_plain, _fc2_sf = _plain_mxfp8_from_bf16(problem)

    torch.testing.assert_close(
        fc1_kernel.view(torch.uint8),
        fc1_plain.view(torch.uint8),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        fc2_kernel.view(torch.uint8),
        fc2_plain.view(torch.uint8),
        atol=0,
        rtol=0,
    )


@pytest.mark.arch_blackwell
def test_mxfp8_preprocess_accepts_sglang_canonical_prequantized_weights():
    _require_cuda()

    import torch

    from flashinfer.moe_ep import MoEWeightPack
    from flashinfer.moe_ep.backends.mega.kernel.sm100.mxfp8_mxfp8_bf16_cutedsl.weights import (
        preprocess_mega_weights,
    )

    problem = _single_rank_problem()
    num_experts = problem["num_experts"]
    hidden, intermediate = problem["hidden"], problem["intermediate"]
    data_dtype = torch.float8_e4m3fn
    Mxfp8ScaleDtype = torch.float8_e8m0fnu
    generator = torch.Generator(device="cuda").manual_seed(19)
    w13 = torch.randn(
        num_experts, 2 * intermediate, hidden, device="cuda", generator=generator
    ).to(data_dtype)
    w2 = torch.randn(
        num_experts, hidden, intermediate, device="cuda", generator=generator
    ).to(data_dtype)
    w13_scale = torch.randint(
        124,
        131,
        (num_experts, 2 * intermediate, hidden // 32),
        device="cuda",
        generator=generator,
        dtype=torch.uint8,
    ).view(Mxfp8ScaleDtype)
    w2_scale = torch.randint(
        124,
        131,
        (num_experts, hidden, intermediate // 32),
        device="cuda",
        generator=generator,
        dtype=torch.uint8,
    ).view(Mxfp8ScaleDtype)

    transformed_l1, transformed_l2 = preprocess_mega_weights(
        MoEWeightPack(
            w13=w13,
            w2=w2,
            w13_scale=w13_scale.view(torch.uint8),
            w2_scale=w2_scale.view(torch.uint8),
        ),
        intermediate_size=intermediate,
        hidden_size=hidden,
        kind=problem["kind"],
    )

    fc1_weight, fc1_sf = transformed_l1
    fc2_weight, fc2_sf = transformed_l2
    assert fc1_weight.shape == (num_experts, hidden, 2 * intermediate)
    assert fc2_weight.shape == (num_experts, intermediate, hidden)
    assert fc1_weight.dtype == data_dtype
    assert fc2_weight.dtype == data_dtype
    assert fc1_sf.dtype == Mxfp8ScaleDtype
    assert fc2_sf.dtype == Mxfp8ScaleDtype

    from flashinfer.moe_ep.shims.sm100.cutedsl_megamoe.sources.kernel_helpers import (
        from_blocked,
    )

    block = 16
    gate_sf, up_sf = w13_scale.chunk(2, dim=1)
    expected_fc1_sf = torch.stack(
        (
            gate_sf.view(torch.uint8).reshape(num_experts, -1, block, hidden // 32),
            up_sf.view(torch.uint8).reshape(num_experts, -1, block, hidden // 32),
        ),
        dim=2,
    ).reshape(num_experts, 2 * intermediate, hidden // 32)
    for expert in range(num_experts):
        torch.testing.assert_close(
            from_blocked(fc1_sf[expert], 2 * intermediate, hidden // 32).view(
                torch.uint8
            ),
            expected_fc1_sf[expert],
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            from_blocked(fc2_sf[expert], hidden, intermediate // 32).view(torch.uint8),
            w2_scale[expert].view(torch.uint8),
            atol=0,
            rtol=0,
        )

    block = 16
    expected_gate0 = w13[:, :block, :].transpose(1, 2).contiguous()
    expected_up0 = (
        w13[:, intermediate : intermediate + block, :].transpose(1, 2).contiguous()
    )
    torch.testing.assert_close(
        fc1_weight[:, :, :block].view(torch.uint8),
        expected_gate0.view(torch.uint8),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        fc1_weight[:, :, block : 2 * block].view(torch.uint8),
        expected_up0.view(torch.uint8),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        fc2_weight.view(torch.uint8),
        w2.transpose(1, 2).contiguous().view(torch.uint8),
        atol=0,
        rtol=0,
    )


@pytest.mark.arch_blackwell
def test_mxfp8_preprocess_and_kernel_match_mega_reference(monkeypatch):
    """Check the public weight transform and standard kernel against Torch terms."""
    _require_cuda()
    import torch

    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("requires SM100/SM103")
    from flashinfer.moe_ep import MoEWeightPack, preprocess_mxfp8_cutedsl_mega_weights
    from flashinfer.moe_ep.shims.sm100.cutedsl_megamoe import sources as pkg
    from flashinfer.moe_ep.shims.sm100.cutedsl_megamoe.sources.kernel_helpers import (
        compute_megamoe_reference_block_scaled,
    )

    monkeypatch.setenv("MEGA_NO_DIST", "1")
    problem = _single_rank_problem()
    weights = preprocess_mxfp8_cutedsl_mega_weights(
        MoEWeightPack(problem["w13"], problem["w2"]),
        intermediate_size=problem["intermediate"],
        hidden_size=problem["hidden"],
        kind=problem["kind"],
    )
    plain1, _, plain2, _ = _plain_mxfp8_from_bf16(problem)
    torch.testing.assert_close(
        weights[0][0].view(torch.uint8), plain1.view(torch.uint8), atol=0, rtol=0
    )
    torch.testing.assert_close(
        weights[1][0].view(torch.uint8), plain2.view(torch.uint8), atol=0, rtol=0
    )
    cfg = pkg.Sm100BlockScaledMoeConfig(
        num_total_experts=problem["num_experts"],
        max_tokens_per_rank=problem["max_tokens"],
        num_topk=problem["topk"],
        hidden=problem["hidden"],
        intermediate=problem["intermediate"],
        rank=0,
        world_size=1,
        quant_kind=problem["kind"],
        gate_up_clamp=problem["gate_up_clamp"],
        apply_routing_weights_before_fc2=True,
    )
    workspace = pkg.Sm100BlockScaledSymmBuffer(cfg)
    tokens = problem["num_tokens"]
    try:
        data, scales = pkg.quantize_mxfp8_block32(
            problem["hidden_states"].float(), torch.float8_e4m3fn
        )
        workspace.x[:tokens].view(torch.uint8).copy_(data.view(torch.uint8))
        workspace.x_sf[:tokens, : scales.shape[1]].view(torch.uint8).copy_(
            scales.view(torch.uint8)
        )
        workspace.topk_idx[:tokens].copy_(problem["topk_ids"])
        workspace.topk_weights[:tokens].copy_(problem["topk_weights"])
        workspace.note_staged_tokens(tokens)
        terms = [
            compute_megamoe_reference_block_scaled(
                workspace.x[:tokens],
                workspace.x_sf[:tokens],
                workspace.topk_idx[:tokens, k : k + 1],
                workspace.topk_weights[:tokens, k : k + 1],
                weights[0][0].permute(0, 2, 1),
                weights[0][1],
                weights[1][0].permute(0, 2, 1),
                weights[1][1],
                quant_kind=problem["kind"],
                local_expert_offset=0,
                gate_up_clamp=problem["gate_up_clamp"],
                apply_routing_weights_before_fc2=True,
                weight_scales_are_swizzled=True,
            )
            for k in range(problem["topk"])
        ]
        pkg.sm100_block_scaled_mega_launch_thunk(*weights, workspace)()
        torch.cuda.synchronize()
        _assert_mega_oracle_term_band_close(
            workspace.output_activation[:tokens],
            torch.stack(terms, dim=1),
            ikr=False,
            label="1gpu",
        )
    finally:
        workspace.destroy()
