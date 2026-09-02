# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused golden for the test-owned SM90 approximate SwiGLU reference."""

from pathlib import Path

import pytest


def test_sm90_swiglu_reference_source_contract_is_independent() -> None:
    helper = Path(__file__).with_name("_sm90_swiglu_reference.py")
    helper_source = helper.read_text(encoding="utf-8")

    assert '"ex2.approx.f32 $0, $1;"' in helper_source
    assert '"rcp.approx.ftz.f32 $0, $1;"' in helper_source
    assert "moe_nvfp4_swapab" not in helper_source
    assert "moe_hopper_fp8" not in helper_source
    assert "torch.exp2" not in helper_source
    assert "torch.reciprocal" not in helper_source
    assert "torch.sigmoid" not in helper_source

    repo_root = Path(__file__).parents[2]
    production_source = (
        repo_root
        / "flashinfer"
        / "moe_ep"
        / "kernel_src"
        / "sm90"
        / "pull_style_cutedsl_megakernel"
        / "src"
        / "moe_hopper_fp8"
        / "epilogue_fp8_common.py"
    ).read_text(encoding="utf-8")
    for needle in (
        "cute.math.exp2(neg_gate_log2e, fastmath=True)",
        "cute.arch.rcp_approx(exp_val + Float32(1.0))",
        "t_up[i] * t_gate[i] * sigmoid * prob",
    ):
        assert needle in production_source


@pytest.mark.arch_hopper
def test_sm90_swiglu_reference_uses_hardware_approximation_boundary() -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("SM90 Hopper is required")

    pytest.importorskip(
        "flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel"
    )
    from tests.moe_ep._sm90_swiglu_reference import swiglu_sm90_reference

    gate = torch.tensor([-2.0, 5.0], dtype=torch.float32, device="cuda")
    up = torch.tensor([3.0, 7.0], dtype=torch.float32, device="cuda")
    actual = swiglu_sm90_reference(gate, up)

    # H200 raw FP32 results for the documented ex2.approx.f32 followed by
    # rcp.approx.ftz.f32 instruction sequence: 0xbf371880, 0x420b1020.
    expected_hardware_bytes = torch.tensor(
        [[0x80, 0x18, 0x37, 0xBF], [0x20, 0x10, 0x0B, 0x42]],
        dtype=torch.uint8,
    )
    assert torch.equal(
        actual.contiguous().view(torch.uint8).cpu().reshape(2, 4),
        expected_hardware_bytes,
    )

    # Keep this vector on a real approximation boundary.  Replacing the tiny
    # PTX reference with precise Torch math produces 0xbf37187f, 0x420b1021.
    precise = (
        up
        * gate
        * torch.reciprocal(
            torch.exp2(-gate * torch.tensor(1.4426950408889634, device="cuda")) + 1.0
        )
    )
    expected_precise_bytes = torch.tensor(
        [[0x7F, 0x18, 0x37, 0xBF], [0x21, 0x10, 0x0B, 0x42]],
        dtype=torch.uint8,
    )
    assert torch.equal(
        precise.contiguous().view(torch.uint8).cpu().reshape(2, 4),
        expected_precise_bytes,
    )
    assert not torch.equal(expected_hardware_bytes, expected_precise_bytes)
