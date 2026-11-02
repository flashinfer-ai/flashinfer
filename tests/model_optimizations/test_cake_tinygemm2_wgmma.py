"""Tests for tinygemm2_wgmma — the generated sm_90a wgmma tinygemm2 programs.

``flashinfer.jit.cake_tinygemm2_wgmma`` holds seven tiles, each in the input
views and cluster sizes the host selects, with and without the bias and with
and without programmatic dependent launch.  Their contract is the output of
``csrc/tinygemm2.cu`` bit for bit, so every parity test below compares the
raw 16-bit patterns (``+0`` and ``-0`` differ), not values within a tolerance.
"""

import os

import pytest
import torch
import torch.nn.functional as F

from flashinfer.utils import get_compute_capability

# (tile, its input views and cluster sizes) as program identifier stems.
STEMS = (
    "PACKED_U4",
    "PACKED_U4_BOX3D",
    "PACKED_S8",
    "PACKED_S8_BOX3D",
    "ACCUMULATORS_CN16",
    "ACCUMULATORS_CN32",
    "ACCUMULATORS_CN32_BOX3D",
    *(f"CLUSTER_S2U4_C{c}" for c in (1, 2, 3, 4)),
    *(f"CLUSTER_S2U4_2D_C{c}" for c in (1, 2, 3, 4)),
    "CLUSTER_S8U8_C4",
    "CLUSTER_R2_C1",
)
PROGRAMS = tuple(
    stem + bias + pdl
    for stem in STEMS
    for bias in ("", "_NOBIAS")
    for pdl in ("", "_PDL")
)

# Batches across the tiles' 8, 16 and 32 rows (ragged tails included), output
# features across their 16, 32 and 64 (ragged tails of the 64-feature tiles).
PARITY_ROWS = [(1, 16), (3, 48), (8, 128), (13, 1040), (16, 2880), (33, 4096)]
# Input features per input view and cluster size: off the 64- and 256-wide K
# steps of the four chains (zero-filled tails), with chains that get no chunk.
# The packed tiles serve multiples of 64 only: one slice, then several rounds.
_INPUTS = {
    "BOX3D": (1024, 4096),
    "C1": (64, 200, 256),
    "C2": (264, 448, 512),
    "C3": (576, 720, 768),
    "C4": (832, 2880, 4160),
    "PACKED": (64, 1088, 2880, 4160),
    "PLAIN": (64, 720, 1040, 4112),
}


def _inputs(program):
    stem = program.removesuffix("_PDL").removesuffix("_NOBIAS")
    if stem.endswith("_BOX3D"):
        return _INPUTS["BOX3D"]
    if stem.startswith("CLUSTER"):
        ks = _INPUTS[stem[-2:]]
        if "_2D_" in stem:
            return tuple(k for k in ks if k % 64 != 0) or (ks[0] + 8,)
        return tuple(k for k in ks if k % 64 == 0)
    if stem.startswith("PACKED"):
        return _INPUTS["PACKED"]
    return _INPUTS["PLAIN"]


CASES = [
    (program, batch, out, k)
    for program in PROGRAMS
    for batch, out in PARITY_ROWS
    for k in _inputs(program)
]


def _require_sm90():
    if not torch.cuda.is_available():
        pytest.skip("tinygemm2_wgmma tests require a CUDA device")
    from flashinfer.jit import cake_tinygemm2_wgmma as programs

    target = programs.PACKED_U4.target_for(get_compute_capability(torch.device("cuda")))
    if target is None:
        pytest.skip("tinygemm2_wgmma requires SM90")
    return target


def _make_case(batch_size, output_features, input_features, seed=0, wide=False):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def operand(*shape):
        values = torch.randn(*shape, generator=g, device="cuda") / 8
        if wide:
            # Exponents from 2^-8 to 2^8: the chains' adds round and cancel.
            values = values * torch.exp2(
                torch.randint(-8, 9, shape, generator=g, device="cuda").float()
            )
        return values.bfloat16()

    return (
        operand(batch_size, input_features),
        operand(output_features, input_features),
        operand(output_features),
    )


def _reference(input, weight, bias, use_pdl):
    from flashinfer.gemm.routergemm import get_tinygemm2_module

    out = torch.zeros(
        input.shape[0], weight.shape[0], device="cuda", dtype=torch.bfloat16
    )
    if bias is None:
        get_tinygemm2_module().tinygemm2_nobias_op(input, weight, out, use_pdl)
    else:
        get_tinygemm2_module().tinygemm2_op(input, weight, bias, out, use_pdl)
    torch.cuda.synchronize()
    return out


@pytest.mark.parametrize("program,batch_size,output_features,input_features", CASES)
@pytest.mark.parametrize("wide", [False, True])
def test_tinygemm2_wgmma_bitwise_parity(
    program, batch_size, output_features, input_features, wide
):
    """Every program is bit-identical to csrc/tinygemm2.cu on every shape it serves."""
    _require_sm90()
    from flashinfer.gemm.routergemm import get_tinygemm2_wgmma_op

    input, weight, bias = _make_case(
        batch_size, output_features, input_features, wide=wide
    )
    if "_NOBIAS" in program:
        bias = None
    use_pdl = program.endswith("_PDL")
    out_ref = _reference(input, weight, bias, use_pdl)
    out = torch.full_like(out_ref, float("nan"))
    get_tinygemm2_wgmma_op()(program, input, weight, bias, out)
    torch.cuda.synchronize()

    bits, bits_ref = out.view(torch.int16), out_ref.view(torch.int16)
    assert torch.equal(bits, bits_ref), (
        f"{program}: bitwise mismatch vs csrc/tinygemm2.cu at batch={batch_size} "
        f"M={output_features} K={input_features}: "
        f"{(bits != bits_ref).sum().item()} differing elements"
    )
    if not wide:
        ref = F.linear(
            input.float(), weight.float(), None if bias is None else bias.float()
        )
        torch.testing.assert_close(out.float(), ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("use_pdl", [False, True])
@pytest.mark.parametrize("has_bias", [False, True])
def test_tinygemm2_wgmma_subnormal_sums(has_bias, use_pdl):
    """Chains that get no input chunk still flush the sum as the reference does:
    a -0 or subnormal sum of the chains that do becomes +0."""
    _require_sm90()
    from flashinfer.gemm.routergemm import get_tinygemm2_wgmma_op

    for program_stem, k in (("CLUSTER_S2U4_C1", 256), ("CLUSTER_S2U4_C3", 704)):
        program = (
            program_stem + ("" if has_bias else "_NOBIAS") + ("_PDL" if use_pdl else "")
        )
        g = torch.Generator(device="cuda").manual_seed(1)
        # Products of 2^-70 and 2^-70: subnormal and -0 sums in FP32.
        input = (
            torch.randn(5, k, generator=g, device="cuda").sign() * 2.0**-70
        ).bfloat16()
        weight = (
            torch.randn(48, k, generator=g, device="cuda").sign() * 2.0**-70
        ).bfloat16()
        bias = (
            torch.zeros(48, device="cuda", dtype=torch.bfloat16) if has_bias else None
        )
        out_ref = _reference(input, weight, bias, use_pdl)
        out = torch.full_like(out_ref, float("nan"))
        get_tinygemm2_wgmma_op()(program, input, weight, bias, out)
        torch.cuda.synchronize()
        assert torch.equal(out.view(torch.int16), out_ref.view(torch.int16)), program


# (batch, output features, input features, bias) -> the program the windows
# select on SM90, or None for the reference kernel.
ROUTES = [
    (1, 128, 720, False, "CLUSTER_S2U4_2D_C3_NOBIAS"),
    (1, 128, 14336, False, "CLUSTER_S8U8_C4_NOBIAS"),
    (2, 16, 256, False, "CLUSTER_S2U4_C1_NOBIAS"),
    (4, 2880, 2880, False, "PACKED_S8_NOBIAS"),
    (7, 128, 4096, False, "CLUSTER_S8U8_C4_NOBIAS"),
    (8, 128, 7168, False, "CLUSTER_S8U8_C4_NOBIAS"),
    (8, 1024, 1024, False, None),
    (13, 1024, 2048, False, "PACKED_U4_BOX3D_NOBIAS"),
    (16, 2880, 2880, False, "ACCUMULATORS_CN16_NOBIAS"),
    (64, 4096, 3072, False, "ACCUMULATORS_CN32_BOX3D_NOBIAS"),
    (16, 1024, 1024, False, "PACKED_U4_BOX3D_NOBIAS"),
    (8, 2048, 2048, False, "PACKED_U4_BOX3D_NOBIAS"),
    (8, 4096, 4096, False, "PACKED_S8_BOX3D_NOBIAS"),
    (16, 4096, 3072, False, "ACCUMULATORS_CN16_NOBIAS"),
    (64, 4096, 4096, False, "ACCUMULATORS_CN32_BOX3D_NOBIAS"),
    (4, 128, 256, False, "CLUSTER_S2U4_C1_NOBIAS"),
    (10, 768, 2000, False, None),
    (13, 1024, 1032, False, None),
    (13, 1024, 1088, False, "PACKED_U4_NOBIAS"),
    (5, 64, 4000, False, None),
    (5, 64, 4032, False, "CLUSTER_S8U8_C4_NOBIAS"),
    (3, 64, 1000, False, None),
    (20, 3072, 5120, False, "ACCUMULATORS_CN16_NOBIAS"),
    (65, 4096, 4096, False, None),
    (64, 4097, 4096, False, None),
    (64, 4096, 14344, False, None),
    (1, 128, 720, True, "CLUSTER_S2U4_2D_C3"),
    (1, 128, 14336, True, "CLUSTER_S8U8_C4"),
    (2, 16, 256, True, "CLUSTER_S2U4_C1"),
    (4, 2880, 2880, True, "PACKED_S8"),
    (7, 128, 4096, True, "CLUSTER_S8U8_C4"),
    (8, 128, 7168, True, "CLUSTER_S8U8_C4"),
    (8, 1024, 1024, True, "PACKED_U4_BOX3D"),
    (13, 1024, 2048, True, "PACKED_U4_BOX3D"),
    (16, 2880, 2880, True, "ACCUMULATORS_CN16"),
    (64, 4096, 3072, True, "ACCUMULATORS_CN32_BOX3D"),
    (16, 1024, 1024, True, "PACKED_U4_BOX3D"),
    (8, 2048, 2048, True, "PACKED_U4_BOX3D"),
    (8, 4096, 4096, True, "ACCUMULATORS_CN16"),
    (16, 4096, 3072, True, "ACCUMULATORS_CN16"),
    (64, 4096, 4096, True, "ACCUMULATORS_CN32_BOX3D"),
    (4, 128, 256, True, "CLUSTER_S2U4_C1"),
    (10, 768, 2000, True, None),
    (13, 1024, 1032, True, None),
    (13, 1024, 1088, True, "PACKED_U4"),
    (5, 64, 4000, True, None),
    (5, 64, 4032, True, "CLUSTER_S8U8_C4"),
    (3, 64, 1000, True, None),
    (20, 3072, 5120, True, "ACCUMULATORS_CN16"),
    (65, 4096, 4096, True, None),
    (64, 4097, 4096, True, None),
    (64, 4096, 14344, True, None),
]


@pytest.mark.parametrize("use_pdl", [False, True])
@pytest.mark.parametrize("batch,out_features,in_features,bias,program", ROUTES)
def test_tinygemm2_wgmma_routes(
    batch, out_features, in_features, bias, program, use_pdl
):
    """PDL selects the same tile through the same windows, as its _PDL program."""
    from flashinfer.gemm.routergemm import _tinygemm2_wgmma_identifier

    expected = None if program is None else program + ("_PDL" if use_pdl else "")
    assert (
        _tinygemm2_wgmma_identifier(batch, out_features, in_features, bias, use_pdl)
        == expected
    )


@pytest.mark.parametrize(
    "batch_size,output_features,input_features",
    [
        (1, 128, 720),
        (2, 16, 256),
        (4, 2880, 2880),
        (8, 128, 7168),
        (8, 1024, 1024),
        (13, 1024, 2048),
        (13, 1024, 1088),
        (16, 1024, 1024),
        (8, 2048, 2048),
        (8, 4096, 4096),
        (16, 2880, 2880),
        (64, 4096, 3072),
    ],
)
@pytest.mark.parametrize("has_bias", [False, True])
def test_tinygemm_bf16_dispatch_matches_reference(
    batch_size, output_features, input_features, has_bias
):
    """The public op returns the reference's bits on SM90 whichever kernel serves it."""
    _require_sm90()
    from flashinfer.gemm import tinygemm_bf16
    from flashinfer.gemm.routergemm import (
        _tinygemm2_wgmma_enabled,
        _tinygemm2_wgmma_program,
    )

    input, weight, bias = _make_case(batch_size, output_features, input_features)
    if not has_bias:
        bias = None
    expected = {route[:4]: route[4] for route in ROUTES}[
        (batch_size, output_features, input_features, has_bias)
    ]
    if not _tinygemm2_wgmma_enabled(
        os.environ.get("FLASHINFER_DISABLE_TINYGEMM2_WGMMA", "0"),
        get_compute_capability(input.device),
    ):
        pytest.skip("tinygemm2_wgmma routing is disabled on this setup")
    assert _tinygemm2_wgmma_program(input, weight, has_bias, False) == expected
    out_ref = _reference(input, weight, bias, False)
    out = torch.full_like(out_ref, float("nan"))
    tinygemm_bf16(input, weight, out, bias)
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.int16), out_ref.view(torch.int16))
