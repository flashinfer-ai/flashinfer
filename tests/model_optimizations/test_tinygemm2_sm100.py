"""Tests for tinygemm2_sm100 — the generated SM100/SM103 tinygemm2 kernel.

``csrc/tinygemm2_sm100.cu`` holds one kernel template instantiated for the
pipeline ring depths 4/8/16 and PDL off/on; it is a generated port of
``csrc/tinygemm2.cu`` whose contract is bit-identical outputs. Every parity
test below therefore uses ``torch.equal``, not a tolerance.
"""

import pytest
import torch
import torch.nn.functional as F

from flashinfer.utils import is_sm100a_supported


def _skip_if_not_sm100_family():
    if not torch.cuda.is_available():
        pytest.skip("tinygemm2_sm100 tests require a CUDA device")
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("tinygemm2_sm100 requires SM100/SM103")


def _make_case(batch_size, output_features, input_features, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    input = (
        torch.randn(
            batch_size, input_features, generator=g, device="cuda", dtype=torch.float32
        )
        / 8
    ).bfloat16()
    weight = (
        torch.randn(
            output_features,
            input_features,
            generator=g,
            device="cuda",
            dtype=torch.float32,
        )
        / 8
    ).bfloat16()
    bias = torch.randn(
        output_features, generator=g, device="cuda", dtype=torch.float32
    ).bfloat16()
    return input, weight, bias


# Ring-depth selection rule of the binding (SelectStages in
# csrc/tinygemm2_sm100.cu), restated here so the tier tests can prove that
# their shapes reach every instantiation on the device they run on:
#   stage4  if K <= 1024 (one loader iteration) or total_ctas > 2 * num_sms;
#   stage16 if K >= 4608 and total_ctas <= num_sms;
#   stage8  otherwise,
# with total_ctas = ceil(M / 16) * ceil(batch / 8).
def _expected_stages(batch_size, output_features, input_features, num_sms):
    total_ctas = ((output_features + 15) // 16) * ((batch_size + 7) // 8)
    if input_features <= 1024 or total_ctas > 2 * num_sms:
        return 4
    if input_features >= 4608 and total_ctas <= num_sms:
        return 16
    return 8


# Shape axes: batch sweeps across the TILE_N=8 boundary (1..7 exercises the
# out-of-bounds TMA box on the batch axis), K sweeps the ring-depth selection
# tiers (K <= 1024 selects the shallow ring; single-wave K >= 4608 selects the
# 16-deep ring), and the large-M rows exercise the grid-size arm of the same
# selection.
PARITY_SHAPES = [
    (1, 128, 720),
    (2, 16, 256),
    (4, 2880, 2880),
    (7, 128, 4096),
    (8, 1024, 1024),
    (8, 128, 7168),
    (13, 1024, 2048),
    (16, 2880, 2880),
    (64, 4096, 3072),
    (1, 128, 14336),
]

# One shape per selection arm. The CTA counts (8, 64, 2048) sit far from the
# SM-count thresholds of every SM100-family part, so the tier each shape
# reaches does not depend on the device; test_tinygemm2_sm100_tiers asserts
# that with the device's actual SM count.
TIER_SHAPES = [
    (1, 128, 720),  # stage4: K fits one loader iteration
    (64, 4096, 3072),  # stage4: 2048 CTAs, multi-wave grid
    (8, 1024, 2048),  # stage8: 64 CTAs, K between the arms
    (8, 128, 7168),  # stage16: 8 CTAs, long K
]


@pytest.mark.parametrize("batch_size,output_features,input_features", PARITY_SHAPES)
@pytest.mark.parametrize("use_pdl", [False, True])
def test_tinygemm2_sm100_bitwise_parity(
    batch_size, output_features, input_features, use_pdl
):
    """The generated kernel must be bit-identical to csrc/tinygemm2.cu."""
    _skip_if_not_sm100_family()
    from flashinfer.gemm.routergemm import (
        get_tinygemm2_module,
        get_tinygemm2_sm100_module,
    )

    input, weight, bias = _make_case(batch_size, output_features, input_features)
    out_ref = torch.zeros(
        batch_size, output_features, device="cuda", dtype=torch.bfloat16
    )
    out_gen = torch.zeros_like(out_ref)

    torch.cuda.synchronize()
    get_tinygemm2_module().tinygemm2_op(input, weight, bias, out_ref, use_pdl)
    torch.cuda.synchronize()
    get_tinygemm2_sm100_module().tinygemm2_sm100_op(
        input, weight, bias, out_gen, use_pdl
    )
    torch.cuda.synchronize()

    assert torch.equal(out_gen, out_ref), (
        f"bitwise mismatch vs csrc/tinygemm2.cu at "
        f"batch={batch_size} M={output_features} K={input_features} pdl={use_pdl}: "
        f"{(out_gen != out_ref).sum().item()} differing elements, "
        f"max |diff|={(out_gen.float() - out_ref.float()).abs().max().item()}"
    )

    ref = F.linear(input.float(), weight.float(), bias.float()).bfloat16()
    torch.testing.assert_close(out_gen.float(), ref.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("use_pdl", [False, True])
def test_tinygemm2_sm100_tiers(use_pdl):
    """Every ring depth is reached through the public op by a shape of
    TIER_SHAPES on this device, and each produces the F.linear result."""
    _skip_if_not_sm100_family()
    from flashinfer.gemm.routergemm import get_tinygemm2_sm100_module

    num_sms = torch.cuda.get_device_properties(
        torch.device("cuda")
    ).multi_processor_count
    tiers = {_expected_stages(*shape, num_sms) for shape in TIER_SHAPES}
    assert tiers == {4, 8, 16}, (
        f"TIER_SHAPES reach ring depths {sorted(tiers)} on a {num_sms}-SM device"
    )

    for batch_size, output_features, input_features in TIER_SHAPES:
        input, weight, bias = _make_case(batch_size, output_features, input_features)
        out = torch.zeros(
            batch_size, output_features, device="cuda", dtype=torch.bfloat16
        )
        get_tinygemm2_sm100_module().tinygemm2_sm100_op(
            input, weight, bias, out, use_pdl
        )
        torch.cuda.synchronize()
        ref = F.linear(input.float(), weight.float(), bias.float()).bfloat16()
        stages = _expected_stages(batch_size, output_features, input_features, num_sms)
        torch.testing.assert_close(
            out.float(),
            ref.float(),
            atol=1e-2,
            rtol=1e-2,
            msg=lambda m: f"stage{stages} pdl={use_pdl} failed: {m}",
        )


@pytest.mark.parametrize("num_launches", [2, 8])
def test_tinygemm2_sm100_pdl_back_to_back(num_launches):
    """PDL launches fired back-to-back must match their eager outputs."""
    _skip_if_not_sm100_family()
    from flashinfer.gemm.routergemm import get_tinygemm2_sm100_module

    batch_size, output_features, input_features = 8, 1024, 2048
    cases = [
        _make_case(batch_size, output_features, input_features, seed=i)
        for i in range(num_launches)
    ]
    outs_eager = []
    for input, weight, bias in cases:
        out = torch.zeros(
            batch_size, output_features, device="cuda", dtype=torch.bfloat16
        )
        get_tinygemm2_sm100_module().tinygemm2_sm100_op(input, weight, bias, out, False)
        torch.cuda.synchronize()
        outs_eager.append(out)

    outs_pdl = [
        torch.zeros(batch_size, output_features, device="cuda", dtype=torch.bfloat16)
        for _ in range(num_launches)
    ]
    torch.cuda.synchronize()
    for (input, weight, bias), out in zip(cases, outs_pdl, strict=True):
        get_tinygemm2_sm100_module().tinygemm2_sm100_op(input, weight, bias, out, True)
    torch.cuda.synchronize()

    for i, (eager, pdl) in enumerate(zip(outs_eager, outs_pdl, strict=True)):
        assert torch.equal(pdl, eager), f"PDL launch {i} diverged from eager output"


def test_tinygemm2_sm100_dispatch_and_escape_hatch(monkeypatch):
    """tinygemm_bf16 must route to the generated backend on SM100/SM103 and
    honor FLASHINFER_DISABLE_TINYGEMM2_SM100, including a toggle at runtime."""
    _skip_if_not_sm100_family()
    import flashinfer.gemm.routergemm as routergemm
    from flashinfer.gemm import tinygemm_bf16

    input, weight, bias = _make_case(4, 128, 720)
    ref = F.linear(input.float(), weight.float(), bias.float()).bfloat16()

    monkeypatch.delenv("FLASHINFER_DISABLE_TINYGEMM2_SM100", raising=False)
    assert routergemm._use_tinygemm2_sm100(input.device)
    out = torch.zeros(4, 128, device="cuda", dtype=torch.bfloat16)
    tinygemm_bf16(input, weight, out, bias=bias)
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), ref.float(), atol=1e-2, rtol=1e-2)

    monkeypatch.setenv("FLASHINFER_DISABLE_TINYGEMM2_SM100", "1")
    assert not routergemm._use_tinygemm2_sm100(input.device)
    out_disabled = torch.zeros(4, 128, device="cuda", dtype=torch.bfloat16)
    tinygemm_bf16(input, weight, out_disabled, bias=bias)
    torch.cuda.synchronize()
    assert torch.equal(out_disabled, out)

    monkeypatch.setenv("FLASHINFER_DISABLE_TINYGEMM2_SM100", "0")
    assert routergemm._use_tinygemm2_sm100(input.device)
