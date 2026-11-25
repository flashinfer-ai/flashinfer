"""CPU tests for the ``mm_fp4`` trace reference and its declared shapes.

The reference must read the inputs the way ``mm_fp4`` reads them. E2M1 codes
are packed two per byte along K. Block scales are in the 128x4-swizzled
buffer layout, and B's scales are the transposed view of that buffer. These
tests build such inputs on CPU and compare the reference with a plain
decode-then-matmul.
"""

import json
from pathlib import Path

import pytest
import torch

FP4_JSON = (
    Path(__file__).resolve().parents[2]
    / "trace/fi_trace_out/gemm_fp4_N2048_K7168_block_size16.json"
)

# E2M1 magnitudes for codes 0..7; bit 3 is the sign.
_E2M1_MAGNITUDES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _decode_e2m1(codes: torch.Tensor) -> torch.Tensor:
    mag = torch.tensor(_E2M1_MAGNITUDES)[codes & 7]
    return torch.where((codes & 8) != 0, -mag, mag)


def _pack_along_k(codes: torch.Tensor) -> torch.Tensor:
    """[rows, K] codes -> [rows, K // 2] uint8, low nibble = even column."""
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)


def _swizzle_128x4(grid: torch.Tensor, fill) -> torch.Tensor:
    """Lay out a [rows, cols] scale grid as the 128x4-swizzled 2D buffer.

    Rows are padded to a multiple of 128 and columns to a multiple of 4. The
    padding cells hold ``fill``, so a reference that reads padding instead of
    trimming it gives a wrong product.
    """
    rows, cols = grid.shape
    padded_rows = (rows + 127) // 128 * 128
    padded_cols = (cols + 3) // 4 * 4
    padded = torch.full((padded_rows, padded_cols), fill, dtype=grid.dtype)
    padded[:rows, :cols] = grid
    r = torch.arange(padded_rows).unsqueeze(1)
    c = torch.arange(padded_cols).unsqueeze(0)
    offset = (
        ((r // 128) * (padded_cols // 4) + c // 4) * 512
        + (r % 32) * 16
        + ((r % 128) // 32) * 4
        + c % 4
    )
    buf = torch.empty(padded_rows * padded_cols, dtype=grid.dtype)
    buf[offset.reshape(-1)] = padded.reshape(-1)
    return buf.reshape(padded_rows, padded_cols)


def _make_case(m: int, n: int, k: int, block_size: int, seed: int = 0):
    """Return (mm_fp4 inputs, expected fp32 product)."""
    gen = torch.Generator().manual_seed(seed)
    a_codes = torch.randint(0, 16, (m, k), generator=gen)
    w_codes = torch.randint(0, 16, (n, k), generator=gen)  # weight [N, K]
    if block_size == 16:
        # NVFP4: E4M3 scales. These values are exact in E4M3.
        choices = torch.tensor([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0])
        a_sf = choices[torch.randint(0, 8, (m, k // 16), generator=gen)]
        w_sf = choices[torch.randint(0, 8, (n, k // 16), generator=gen)]
        a_buf = _swizzle_128x4(a_sf, 448.0).to(torch.float8_e4m3fn)
        w_buf = _swizzle_128x4(w_sf, 448.0).to(torch.float8_e4m3fn)
    else:
        # MXFP4: UE8M0 scale bytes, value = 2 ** (byte - 127).
        a_exp = torch.randint(124, 131, (m, k // 32), generator=gen)
        w_exp = torch.randint(124, 131, (n, k // 32), generator=gen)
        a_sf = torch.pow(2.0, a_exp.float() - 127)
        w_sf = torch.pow(2.0, w_exp.float() - 127)
        a_buf = _swizzle_128x4(a_exp.to(torch.uint8), 200)
        w_buf = _swizzle_128x4(w_exp.to(torch.uint8), 200)
    a = _decode_e2m1(a_codes) * a_sf.repeat_interleave(block_size, dim=1)
    w = _decode_e2m1(w_codes) * w_sf.repeat_interleave(block_size, dim=1)
    inputs = dict(
        A=_pack_along_k(a_codes),  # [M, K // 2]
        B=_pack_along_k(w_codes).T,  # [K // 2, N], the view mm_fp4 takes
        a_descale=a_buf,
        b_descale=w_buf.T,  # transposed view of B's scale buffer
        block_size=block_size,
    )
    return inputs, a @ w.T


@pytest.mark.cpu_only
@pytest.mark.parametrize("block_size", [16, 32])
@pytest.mark.parametrize("m,n,k", [(128, 128, 64), (3, 256, 96), (130, 64, 128)])
def test_mm_fp4_reference_matches_dequantized_matmul(m, n, k, block_size):
    from flashinfer.trace.templates.gemm import mm_fp4_trace

    inputs, expected = _make_case(m, n, k, block_size)
    out = mm_fp4_trace.reference(**inputs)
    assert out.shape == (m, n)
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), expected, rtol=1e-2, atol=1e-2)


@pytest.mark.cpu_only
def test_mm_fp4_reference_reads_the_sign_bit():
    from flashinfer.trace.templates.gemm import mm_fp4_trace

    m, n, k = 1, 2, 16
    ones = torch.ones(1, 1)
    inputs = dict(
        A=_pack_along_k(torch.full((m, k), 0xC)),  # -2.0
        B=_pack_along_k(torch.full((n, k), 0x6)).T,  # 4.0
        a_descale=_swizzle_128x4(ones, 1.0).to(torch.float8_e4m3fn),
        b_descale=_swizzle_128x4(ones.expand(n, 1), 1.0).to(torch.float8_e4m3fn).T,
    )
    out = mm_fp4_trace.reference(**inputs)
    assert torch.equal(out.float(), torch.full((m, n), -2.0 * 4.0 * k))


@pytest.mark.cpu_only
@pytest.mark.skipif(
    not hasattr(torch, "float4_e2m1fn_x2"), reason="torch has no float4_e2m1fn_x2"
)
def test_mm_fp4_reference_accepts_float4_dtype():
    """mm_fp4 also takes A and B as float4_e2m1fn_x2; the reference reads them
    as the same bytes."""
    from flashinfer.trace.templates.gemm import mm_fp4_trace

    inputs, expected = _make_case(130, 64, 128, 16, seed=3)
    as_fp4 = dict(
        inputs,
        A=inputs["A"].view(torch.float4_e2m1fn_x2),
        B=inputs["B"].view(torch.float4_e2m1fn_x2),
    )
    out = mm_fp4_trace.reference(**as_fp4).float()
    torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)


# Byte offset at which nvfp4_quantize(..., SfLayout.layout_128x4) writes the
# scale of block (row, column), as recorded from the library. The input is
# 130 x 128: 8 scale columns, and rows padded to 256. The rows and columns
# cross every boundary of the 128x4 layout (32 rows, 128 rows, 4 columns).
_QUANTIZER_SCALE_OFFSETS = {
    (0, 0): 0,
    (0, 3): 3,
    (0, 4): 512,
    (0, 7): 515,
    (1, 0): 16,
    (1, 3): 19,
    (1, 4): 528,
    (1, 7): 531,
    (31, 0): 496,
    (31, 3): 499,
    (31, 4): 1008,
    (31, 7): 1011,
    (32, 0): 4,
    (32, 3): 7,
    (32, 4): 516,
    (32, 7): 519,
    (33, 0): 20,
    (33, 3): 23,
    (33, 4): 532,
    (33, 7): 535,
    (63, 0): 500,
    (63, 3): 503,
    (63, 4): 1012,
    (63, 7): 1015,
    (64, 0): 8,
    (64, 3): 11,
    (64, 4): 520,
    (64, 7): 523,
    (127, 0): 508,
    (127, 3): 511,
    (127, 4): 1020,
    (127, 7): 1023,
    (128, 0): 1024,
    (128, 3): 1027,
    (128, 4): 1536,
    (128, 7): 1539,
    (129, 0): 1040,
    (129, 3): 1043,
    (129, 4): 1552,
    (129, 7): 1555,
}


@pytest.mark.cpu_only
def test_unswizzle_matches_quantizer_placement():
    """_unswizzle_sf_128x4 (used by the reference) finds each scale where the
    quantizer writes it. This checks more than agreement with _swizzle_128x4."""
    from flashinfer.trace.templates.gemm import _unswizzle_sf_128x4

    buf = torch.zeros(256 * 8, dtype=torch.uint8)
    expected = torch.zeros(130, 8, dtype=torch.uint8)
    for i, ((r, c), offset) in enumerate(sorted(_QUANTIZER_SCALE_OFFSETS.items())):
        buf[offset] = i + 1
        expected[r, c] = i + 1
    assert torch.equal(_unswizzle_sf_128x4(buf, 130, 8), expected)


@pytest.mark.cpu_only
def test_unpack_e2m1_matches_library_cpu_decoder():
    """The reference's E2M1 decode agrees with fp4_quantization's own CPU decoder,
    which the reference does not share code with."""
    from flashinfer.quantization.fp4_quantization import (
        _e2m1_and_ufp8sf_scale_to_float_cpu,
    )
    from flashinfer.trace.templates.gemm import _unpack_e2m1

    codes = torch.arange(64).remainder(16).reshape(2, 32)
    packed = _pack_along_k(codes)
    unit_scales = torch.full((2, 1), 127, dtype=torch.uint8)  # UE8M0 2**0
    expected = _e2m1_and_ufp8sf_scale_to_float_cpu(
        packed, unit_scales, None, 32, 0, is_sf_swizzled_layout=False
    )
    torch.testing.assert_close(_unpack_e2m1(packed), expected.float(), atol=0, rtol=0)


@pytest.mark.cpu_only
def test_mm_fp4_reference_reads_float8_typed_mxfp4_scales():
    """mm_fp4 accepts MXFP4 scales typed float8_e4m3fn as well as uint8. It reads
    the same UE8M0 bytes, so the reference gives the same result for both."""
    from flashinfer.trace.templates.gemm import mm_fp4_trace

    inputs, expected = _make_case(130, 64, 128, 32, seed=7)
    as_float8 = dict(
        inputs,
        a_descale=inputs["a_descale"].view(torch.float8_e4m3fn),
        b_descale=inputs["b_descale"].view(torch.float8_e4m3fn),
    )
    out = mm_fp4_trace.reference(**as_float8).float()
    torch.testing.assert_close(out, mm_fp4_trace.reference(**inputs).float())
    torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.cpu_only
def test_mm_fp4_reference_applies_alpha():
    """alpha (NVFP4's global scale) multiplies the output, as in mm_fp4.

    Also checks that a per-token alpha and scales of another dtype are
    refused.
    """
    from flashinfer.trace.templates.gemm import mm_fp4_trace

    inputs, expected = _make_case(130, 64, 128, 16, seed=5)
    alpha = torch.tensor([0.25])
    out = mm_fp4_trace.reference(**inputs, alpha=alpha).float()
    torch.testing.assert_close(out, expected * 0.25, rtol=1e-2, atol=1e-2)

    # The template declares one global alpha, so a per-token alpha is refused.
    with pytest.raises(ValueError, match="one global alpha"):
        mm_fp4_trace.reference(**inputs, alpha=torch.ones(130))
    # Scales of another dtype are refused, not reinterpreted.
    bad = dict(inputs, a_descale=inputs["a_descale"].float())
    with pytest.raises(ValueError, match="scales must be uint8 or float8_e4m3fn"):
        mm_fp4_trace.reference(**bad)


@pytest.mark.cpu_only
def test_mm_fp4_json_reference_matches_source():
    """The committed JSON must be the definition the template renders now.

    The whole definition is compared (reference, init, axes, inputs and the
    rest), so a template change cannot leave a stale copy behind.
    """
    import flashinfer

    m, k, n, bs = 128, 7168, 2048, 16
    definition = flashinfer.gemm.mm_fp4.fi_trace(
        a=torch.zeros(m, k // 2, dtype=torch.uint8),
        b=torch.zeros(n, k // 2, dtype=torch.uint8).T,
        a_descale=torch.ones(m, k // bs, dtype=torch.float8_e4m3fn),
        b_descale=torch.ones(n, k // bs, dtype=torch.float8_e4m3fn).T,
        block_size=bs,
    )
    committed = json.loads(FP4_JSON.read_text())
    assert committed == json.loads(json.dumps(definition))


@pytest.mark.cpu_only
def test_mm_fp4_json_reference_runs_standalone():
    """The committed JSON's reference must exec without flashinfer and agree."""
    doc = json.loads(FP4_JSON.read_text())
    namespace: dict = {}
    exec(doc["reference"], namespace)  # noqa: S102
    inputs, expected = _make_case(3, 256, 96, 16)
    out = namespace["_mm_fp4_reference"](**inputs)
    torch.testing.assert_close(out.float(), expected, rtol=1e-2, atol=1e-2)


@pytest.mark.cpu_only
def test_mm_fp4_trace_reads_k_and_n_from_packed_inputs():
    import flashinfer

    m, n, k = 3, 256, 512
    inputs, _ = _make_case(m, n, k, 16)
    defn = flashinfer.gemm.mm_fp4.fi_trace(
        a=inputs["A"],
        b=inputs["B"],
        a_descale=inputs["a_descale"],
        b_descale=inputs["b_descale"],
        block_size=16,
    )
    assert defn["axes"]["K"]["value"] == k
    assert defn["axes"]["N"]["value"] == n
    assert defn["name"] == f"gemm_fp4_N{n}_K{k}_block_size16"
