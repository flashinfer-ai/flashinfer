"""Tests for tinygemm2_sm100 — the generated SM100/SM103 tinygemm2 variants.

The four frozen variants in ``csrc/tinygemm2_sm100/tinygemm2_sm100.cu`` are
generated Loom schedules exactly porting ``csrc/tinygemm2.cu``; the contract
is bit-identical outputs. Every parity test below therefore uses
``torch.equal``, not a tolerance.
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


# Shape axes: batch sweeps across the TILE_N=8 boundary (1..7 exercises the
# out-of-bounds TMA box on the batch axis), K sweeps both sides of the
# stage4/stage8 selection point (K <= 1024 selects the shallow ring), and the
# large-M row exercises the grid-size arm of the same selection.
PARITY_SHAPES = [
    (1, 128, 720),
    (2, 16, 256),
    (4, 2880, 2880),
    (7, 128, 4096),
    (8, 1024, 1024),
    (13, 1024, 2048),
    (16, 2880, 2880),
    (64, 4096, 3072),
]


@pytest.mark.parametrize("batch_size,output_features,input_features", PARITY_SHAPES)
@pytest.mark.parametrize("use_pdl", [False, True])
def test_tinygemm2_sm100_bitwise_parity(
    batch_size, output_features, input_features, use_pdl
):
    """The generated variants must be bit-identical to csrc/tinygemm2.cu."""
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


@pytest.mark.parametrize(
    "variant_op,smem_note",
    [
        ("stage8_op", "deep ring"),
        ("stage8_pdl_op", "deep ring + PDL"),
        ("stage4_op", "shallow ring"),
        ("stage4_pdl_op", "shallow ring + PDL"),
    ],
)
def test_tinygemm2_sm100_each_variant(variant_op, smem_note):
    """Drive each frozen variant directly, bypassing the stage selection."""
    _skip_if_not_sm100_family()
    from flashinfer.jit import gen_tinygemm2_sm100_module

    module = gen_tinygemm2_sm100_module().build_and_load()
    op = getattr(module, variant_op)

    for batch_size, output_features, input_features in [(1, 128, 720), (8, 1024, 2048)]:
        input, weight, bias = _make_case(batch_size, output_features, input_features)
        out = torch.zeros(
            batch_size, output_features, device="cuda", dtype=torch.bfloat16
        )
        op(input, weight, bias, out)
        torch.cuda.synchronize()
        ref = F.linear(input.float(), weight.float(), bias.float()).bfloat16()
        torch.testing.assert_close(
            out.float(),
            ref.float(),
            atol=1e-2,
            rtol=1e-2,
            msg=lambda m: f"{variant_op} ({smem_note}) failed: {m}",
        )


@pytest.mark.parametrize("num_launches", [2, 8])
def test_tinygemm2_sm100_pdl_back_to_back(num_launches):
    """PDL variants fired back-to-back must match their eager outputs."""
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
    honor FLASHINFER_DISABLE_TINYGEMM2_SM100."""
    _skip_if_not_sm100_family()
    import flashinfer.gemm.routergemm as routergemm
    from flashinfer.gemm import tinygemm_bf16

    input, weight, bias = _make_case(4, 128, 720)
    ref = F.linear(input.float(), weight.float(), bias.float()).bfloat16()

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
