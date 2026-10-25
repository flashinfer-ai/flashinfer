"""Tests for tinygemm2_tcgen05 — the generated tcgen05 tinygemm2 programs.

``flashinfer.jit.cake_tinygemm2_tcgen05`` holds three tiles, each with and
without the bias and with and without programmatic dependent launch.  Their
contract is the output of ``csrc/tinygemm2.cu`` bit for bit, so every parity
test below uses ``torch.equal``, not a tolerance.
"""

import pytest
import torch
import torch.nn.functional as F

from flashinfer.utils import get_compute_capability

TILES = ("PACKED_NARROW", "PACKED", "ACCUMULATORS")
PROGRAMS = tuple(
    tile + bias + pdl
    for tile in TILES
    for bias in ("", "_NOBIAS")
    for pdl in ("", "_PDL")
)

# Every program runs every shape: batches across the tiles' 2, 16 and 64 rows,
# output features across their 8 and 32, and input features off the 64- and
# 256-wide K steps of the four chains (zero-filled tails), with chains that
# get no K chunk at all (K <= 768).
PARITY_SHAPES = [
    (1, 16, 64),
    (1, 128, 720),
    (2, 16, 256),
    (3, 48, 4112),
    (5, 80, 1040),
    (7, 128, 4096),
    (8, 128, 7168),
    (9, 2880, 2880),
    (16, 4096, 3072),
    (33, 48, 528),
    (40, 2880, 2880),
    (64, 4096, 3072),
]


def _target():
    if not torch.cuda.is_available():
        pytest.skip("tinygemm2_tcgen05 tests require a CUDA device")
    from flashinfer.jit import cake_tinygemm2_tcgen05 as programs

    target = programs.PACKED.target_for(get_compute_capability(torch.device("cuda")))
    if target is None:
        pytest.skip("tinygemm2_tcgen05 requires SM100, SM103 or SM107")
    if target == "sm_107a":
        from flashinfer.jit.cpp_ext import is_cuda_version_at_least

        if not is_cuda_version_at_least("13.4"):
            pytest.skip("the sm_107a programs require CUDA 13.4")
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


@pytest.mark.parametrize("batch_size,output_features,input_features", PARITY_SHAPES)
@pytest.mark.parametrize("program", PROGRAMS)
@pytest.mark.parametrize("wide", [False, True])
def test_tinygemm2_tcgen05_bitwise_parity(
    batch_size, output_features, input_features, program, wide
):
    """Every program is bit-identical to csrc/tinygemm2.cu on every shape."""
    _target()
    from flashinfer.gemm.routergemm import get_tinygemm2_tcgen05_op

    input, weight, bias = _make_case(
        batch_size, output_features, input_features, wide=wide
    )
    if "_NOBIAS" in program:
        bias = None
    use_pdl = program.endswith("_PDL")
    out_ref = _reference(input, weight, bias, use_pdl)
    out = torch.full_like(out_ref, float("nan"))
    get_tinygemm2_tcgen05_op()(program, input, weight, bias, out)
    torch.cuda.synchronize()

    assert torch.equal(out, out_ref), (
        f"{program}: bitwise mismatch vs csrc/tinygemm2.cu at batch={batch_size} "
        f"M={output_features} K={input_features}: "
        f"{(out != out_ref).sum().item()} differing elements"
    )
    if not wide:
        ref = F.linear(
            input.float(), weight.float(), None if bias is None else bias.float()
        )
        torch.testing.assert_close(out.float(), ref, atol=1e-2, rtol=1e-2)


# (capability, batch, output features, input features, bias) -> the tile the
# windows select, or None for the kernel the call runs otherwise.
ROUTES = [
    ((10, 0), 7, 128, 4096, True, "PACKED_NARROW"),
    ((10, 0), 1, 128, 14336, True, "PACKED_NARROW"),
    ((10, 0), 8, 256, 4096, False, "PACKED_NARROW"),
    ((10, 0), 8, 272, 4096, False, None),
    ((10, 0), 8, 128, 4080, True, None),
    ((10, 0), 4, 2880, 2880, False, "PACKED"),
    ((10, 0), 4, 2880, 2880, True, None),
    ((10, 0), 9, 2880, 2880, True, "PACKED"),
    ((10, 0), 32, 4096, 3072, True, "PACKED"),
    ((10, 0), 33, 4096, 3072, True, "ACCUMULATORS"),
    ((10, 0), 64, 4096, 4096, False, "ACCUMULATORS"),
    ((10, 0), 65, 4096, 4096, False, None),
    ((10, 0), 16, 2048, 2048, True, None),
    ((10, 0), 8, 1024, 1024, False, None),
    ((10, 3), 7, 128, 4096, True, "PACKED_NARROW"),
    ((10, 3), 8, 128, 7168, True, None),
    ((10, 3), 1, 128, 14336, True, None),
    ((10, 3), 7, 128, 4096, False, None),
    ((10, 3), 8, 128, 7168, False, "PACKED_NARROW"),
    ((10, 3), 1, 128, 14336, False, "PACKED_NARROW"),
    ((10, 3), 8, 4096, 4096, False, "PACKED"),
    ((10, 3), 8, 4096, 4096, True, None),
    ((10, 3), 16, 2880, 2880, True, "PACKED"),
    ((10, 3), 64, 4096, 3072, True, "ACCUMULATORS"),
    ((10, 3), 13, 1024, 2048, True, None),
    ((10, 7), 64, 4096, 3072, True, None),
    ((9, 0), 64, 4096, 3072, True, None),
]


@pytest.mark.parametrize(
    "capability,batch_size,output_features,input_features,has_bias,tile", ROUTES
)
@pytest.mark.parametrize("use_pdl", [False, True])
def test_tinygemm2_tcgen05_windows(
    monkeypatch,
    capability,
    batch_size,
    output_features,
    input_features,
    has_bias,
    tile,
    use_pdl,
):
    """The windows select the tile of each shape per compute capability."""
    import flashinfer.gemm.routergemm as routergemm

    monkeypatch.setattr(routergemm, "get_compute_capability", lambda _: capability)
    monkeypatch.delenv("FLASHINFER_DISABLE_TINYGEMM2_TCGEN05", raising=False)
    input = torch.empty(batch_size, input_features, dtype=torch.bfloat16)
    weight = torch.empty(output_features, input_features, dtype=torch.bfloat16)
    program = routergemm._tinygemm2_tcgen05_program(input, weight, has_bias, use_pdl)
    expected = (
        None
        if tile is None
        else tile + ("" if has_bias else "_NOBIAS") + ("_PDL" if use_pdl else "")
    )
    assert program == expected


def test_tinygemm2_tcgen05_disabled_and_unaligned(monkeypatch):
    """The escape hatch and a tensor-map-unaligned operand keep the kernels the
    call runs otherwise."""
    import flashinfer.gemm.routergemm as routergemm

    monkeypatch.setattr(routergemm, "get_compute_capability", lambda _: (10, 0))
    input = torch.empty(64, 4096, dtype=torch.bfloat16)
    weight = torch.empty(4096, 4096, dtype=torch.bfloat16)
    assert routergemm._tinygemm2_tcgen05_program(input, weight, True, False)
    monkeypatch.setenv("FLASHINFER_DISABLE_TINYGEMM2_TCGEN05", "1")
    assert routergemm._tinygemm2_tcgen05_program(input, weight, True, False) is None
    monkeypatch.delenv("FLASHINFER_DISABLE_TINYGEMM2_TCGEN05")
    storage = torch.empty(64 * 4096 + 1, dtype=torch.bfloat16)
    shifted = storage[1:].view(64, 4096)
    assert routergemm._tinygemm2_tcgen05_program(shifted, weight, True, False) is None
    assert (
        routergemm._tinygemm2_tcgen05_program(
            torch.empty(64, 4100, dtype=torch.bfloat16),
            torch.empty(4096, 4100, dtype=torch.bfloat16),
            True,
            False,
        )
        is None
    )


@pytest.mark.parametrize(
    "batch_size,output_features,input_features",
    sorted({route[1:4] for route in ROUTES}),
)
@pytest.mark.parametrize("has_bias", [True, False])
@pytest.mark.parametrize("use_pdl", [False, True])
def test_tinygemm_bf16_routes_bitwise(
    batch_size, output_features, input_features, has_bias, use_pdl
):
    """The public op returns the reference's bits whichever kernel it routes to."""
    _target()
    from flashinfer.gemm.routergemm import tinygemm_bf16

    input, weight, bias = _make_case(batch_size, output_features, input_features)
    if not has_bias:
        bias = None
    out_ref = _reference(input, weight, bias, use_pdl)
    out = torch.full_like(out_ref, float("nan"))
    tinygemm_bf16(input, weight, out, bias=bias, use_pdl=use_pdl)
    torch.cuda.synchronize()
    assert torch.equal(out, out_ref)
