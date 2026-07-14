"""Single-GPU checks: FlashInfer MXFP8 preprocess vs ``compute_megamoe_reference_mxfp8``.

Validates that ``mxfp8_cutedsl.preprocess_mega_weights`` produces fp8 weights and
plain E8M0 scale layouts consistent with the CuTeDSL torch reference, and that a
single-rank ``mxfp8_mega_moe`` launch matches the reference after weighted top-k
reduction.

Run on one Blackwell GPU from the FlashInfer repo root (no torchrun required)::

    cd /path/to/flashinfer
    export PYTHONPATH="${PWD}:${PYTHONPATH}"
    MEGA_NO_DIST=1 CUDA_VISIBLE_DEVICES=0 pytest \\
        tests/moe_ep/test_mxfp8_cutedsl_preprocess_vs_reference.py -v \\
        -m arch_blackwell --confcutdir=tests/moe_ep
"""

from __future__ import annotations

import pytest

pytest.importorskip("flashinfer.moe_ep.backends.mega.kernel.cutedsl_backend_kernels")


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
    from flashinfer.moe_ep.backends.mega.kernel.mxfp8_cutedsl.weights import (
        _fc1_weight_from_w13,
        _quantize_mxfp8_weight_k_major,
    )

    intermediate_size = problem["intermediate"]
    kind = problem["kind"]
    pack = MoEWeightPack(w13=problem["w13"], w2=problem["w2"])
    num_experts = pack.w13.shape[0]

    fc1_fp32 = _fc1_weight_from_w13(pack.w13, intermediate_size=intermediate_size)
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


@pytest.mark.arch_blackwell
def test_mxfp8_preprocess_fp8_weights_match_plain_quant():
    """``preprocess_mega_weights`` fp8 tensors match an independent plain quant."""
    _require_cuda()

    import torch

    from flashinfer.moe_ep import MoEWeightPack
    from flashinfer.moe_ep.backends.mega.kernel.mxfp8_cutedsl.weights import (
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
    from flashinfer.moe_ep.backends.mega.kernel.mxfp8_cutedsl.weights import (
        preprocess_mega_weights,
    )
    from common.megamoe_constants import Mxfp8BlockSize
    from moe_mxfp8_glu.mega_runner import (
        _make_e8m0_scale_tensor,
        _make_fp8_tensor,
    )
    from moe_nvfp4_swapab.runner_common import Mxfp8ScaleDtype

    problem = _single_rank_problem()
    num_experts = problem["num_experts"]
    hidden = problem["hidden"]
    intermediate = problem["intermediate"]
    data_dtype = torch.float8_e4m3fn

    g = torch.Generator(device="cuda").manual_seed(19)
    w13 = _make_fp8_tensor(
        g,
        (num_experts, 2 * intermediate, hidden),
        data_dtype,
        perf_run=True,
    )
    w2 = _make_fp8_tensor(
        g,
        (num_experts, hidden, intermediate),
        data_dtype,
        perf_run=True,
    )
    w13_scale = _make_e8m0_scale_tensor(
        g,
        num_experts * 2 * intermediate,
        hidden,
        blocksize=Mxfp8BlockSize,
    ).reshape(num_experts, 2 * intermediate, hidden // Mxfp8BlockSize)
    w2_scale = _make_e8m0_scale_tensor(
        g,
        num_experts * hidden,
        intermediate,
        blocksize=Mxfp8BlockSize,
    ).reshape(num_experts, hidden, intermediate // Mxfp8BlockSize)

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

    block = Mxfp8BlockSize
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
    """Single-rank kernel output matches ``compute_megamoe_reference_mxfp8``."""
    _require_cuda()

    import torch

    cap = torch.cuda.get_device_capability()
    if cap[0] != 10:
        pytest.skip(
            f"mxfp8_mega_moe requires sm_100a or sm_103a; got sm_{cap[0]}{cap[1]}"
        )

    from flashinfer.moe_ep.backends.mega.kernel.cutedsl_backend_kernels.frontend import (
        get_symm_buffer_for_mxfp8_mega_moe,
        mxfp8_mega_moe,
    )
    from flashinfer.moe_ep import MoEWeightPack
    from flashinfer.moe_ep.backends.mega.kernel.mxfp8_cutedsl.staging import (
        stage_mega_moe_inputs,
    )
    from flashinfer.moe_ep.backends.mega.kernel.mxfp8_cutedsl.weights import (
        preprocess_mega_weights,
    )
    from moe_mxfp8_glu.mega_reference_mxfp8 import compute_megamoe_reference_mxfp8

    # monkeypatch (not os.environ): restored after the test, so it cannot
    # silently downgrade later nvshmem-path tests in the same process.
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    problem = _single_rank_problem()
    rank = 0
    world_size = 1
    num_tokens = problem["num_tokens"]
    kind = problem["kind"]
    data_dtype = torch.float8_e4m3fn

    pack = MoEWeightPack(w13=problem["w13"], w2=problem["w2"])
    transformed_l1, transformed_l2 = preprocess_mega_weights(
        pack,
        intermediate_size=problem["intermediate"],
        hidden_size=problem["hidden"],
        kind=kind,
        gate_up_clamp=problem["gate_up_clamp"],
    )

    fc1_kernel, fc1_plain_sf, fc2_kernel, fc2_plain_sf = _plain_mxfp8_from_bf16(problem)
    _pre_fc1, _pre_fc2 = transformed_l1[0], transformed_l2[0]
    torch.testing.assert_close(
        _pre_fc1.view(torch.uint8), fc1_kernel.view(torch.uint8), atol=0, rtol=0
    )
    torch.testing.assert_close(
        _pre_fc2.view(torch.uint8), fc2_kernel.view(torch.uint8), atol=0, rtol=0
    )

    symm_buffer = get_symm_buffer_for_mxfp8_mega_moe(
        problem["num_experts"],
        problem["max_tokens"],
        problem["topk"],
        problem["hidden"],
        problem["intermediate"],
        rank,
        world_size,
        kind=kind,
        gate_up_clamp=problem["gate_up_clamp"],
    )
    try:
        stage_mega_moe_inputs(
            problem["hidden_states"],
            problem["topk_weights"],
            problem["topk_ids"],
            symm_buffer.x,
            symm_buffer.x_sf,
            symm_buffer.topk_idx,
            symm_buffer.topk_weights,
            kind=kind,
        )

        act = symm_buffer.x[:num_tokens]
        act_sf = symm_buffer.x_sf[:num_tokens]
        topk_idx = symm_buffer.topk_idx[:num_tokens]
        topk_weights = symm_buffer.topk_weights[:num_tokens]

        combine_ref = compute_megamoe_reference_mxfp8(
            input_activation=act.unsqueeze(0),
            input_activation_sf=act_sf.unsqueeze(0),
            input_topk_idx=topk_idx.unsqueeze(0),
            input_topk_weights=topk_weights.unsqueeze(0),
            fc1_weight=fc1_kernel.unsqueeze(0),
            fc1_weight_sf=fc1_plain_sf.unsqueeze(0),
            fc2_weight=fc2_kernel.unsqueeze(0),
            fc2_weight_sf=fc2_plain_sf.unsqueeze(0),
            ab_dtype=data_dtype,
            gate_up_clamp=problem["gate_up_clamp"],
        )
        y_ref = (
            (
                combine_ref[0].to(torch.float32)
                * topk_weights[:, :, None].to(torch.float32)
            )
            .sum(dim=1)
            .to(torch.bfloat16)
        )

        y_kernel = torch.empty(
            num_tokens, problem["hidden"], dtype=torch.bfloat16, device="cuda"
        )
        mxfp8_mega_moe(
            y_kernel,
            transformed_l1,
            transformed_l2,
            symm_buffer,
            num_tokens=num_tokens,
            gate_up_clamp=problem["gate_up_clamp"],
        )
        torch.cuda.synchronize()

        # Per-(token, topk) cells first (Form A); then the host top-k reduction.
        # Random bf16 activations/weights yield |y|~1e2–1e3; kernel vs torch
        # ref can differ by ~1 bf16 ULP (|Δ|≈8) on a handful of cells.
        _atol = 8.0
        _rtol = 0.05
        y_kernel_per_topk = symm_buffer.combine_output[:num_tokens].to(torch.float32)
        y_ref_per_topk = combine_ref[0].to(torch.float32)
        torch.testing.assert_close(
            y_kernel_per_topk,
            y_ref_per_topk,
            atol=_atol,
            rtol=_rtol,
        )
        torch.testing.assert_close(
            y_kernel.to(torch.float32),
            y_ref.to(torch.float32),
            atol=_atol,
            rtol=_rtol,
        )
    finally:
        symm_buffer.destroy()
