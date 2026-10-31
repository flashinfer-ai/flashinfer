"""Tests for tinygemm2_tcgen05 — the generated tcgen05 tinygemm2 programs.

``flashinfer.jit.cake_tinygemm2_tcgen05`` holds nine tiles, each with and
without the bias and with and without programmatic dependent launch.  Their
contract is the output of ``csrc/tinygemm2.cu`` bit for bit, so every parity
test below uses ``torch.equal``, not a tolerance.
"""

import pytest
import torch
import torch.nn.functional as F

from flashinfer.utils import get_compute_capability

TILES = (
    "PACKED_NARROW",
    "PACKED_NARROW_U4",
    "PACKED_NARROW_FULL_STAGES",
    "PACKED_M64_N16",
    "PACKED_M64_N4",
    "PACKED",
    "PACKED_BOX8",
    "PACKED_M64_N8",
    "ACCUMULATORS",
)
PROGRAMS = tuple(
    tile + bias + pdl
    for tile in TILES
    for bias in ("", "_NOBIAS")
    for pdl in ("", "_PDL")
)

# Every program runs every shape: batches across the tiles' 2, 4, 8, 16 and 64 rows,
# output features across their 8, 16 and 32, and input features off the 64- and
# 256-wide K steps of the four chains (zero-filled tails), with chains that
# get no K chunk at all (K <= 768).
PARITY_SHAPES = [
    (1, 16, 64),
    (1, 128, 720),
    (2, 16, 256),
    (3, 48, 4112),
    (4, 2880, 2880),
    (5, 80, 1040),
    (7, 128, 4096),
    (8, 128, 7168),
    (8, 1040, 4160),
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


# (batch, output features, input features, bias) -> the tile the windows select
# on sm_100a and sm_103a, or None for the kernel the call runs otherwise.  The
# one-wave rule is checked at 148 SMs.
SM100_FAMILY_ROUTES = [
    # PACKED up to eight rows: no bias, 2880 to 4096 output features, K at most 6144
    (1, 2880, 2112, False, "PACKED"),
    (1, 2880, 2120, False, "PACKED"),
    (4, 2880, 2880, False, "PACKED"),
    (8, 4096, 4096, False, "PACKED"),
    (8, 2864, 4096, False, None),
    (2, 2048, 3072, False, None),
    (7, 4112, 2056, False, None),
    (8, 3072, 6152, False, None),
    (4, 2880, 2880, True, None),
    (8, 4096, 4096, True, None),
    # PACKED with a bias on 9 to 16 rows: from 2880 output features and K 2112
    (9, 2880, 2112, True, "PACKED"),
    (16, 2880, 2880, True, "PACKED"),
    (16, 4096, 3072, True, "PACKED"),
    (14, 3072, 2048, True, None),
    (9, 2048, 3072, True, None),
    (17, 2880, 2880, True, None),
    (24, 3072, 2048, True, None),
    # from 9 rows without a bias, 28 with one (from 1536 output features up to
    # 32 rows, from 1024 past that): PACKED while its grid fits one wave
    (9, 2048, 3072, False, "PACKED"),
    (16, 2880, 2880, False, "PACKED"),
    (24, 2048, 3072, False, "PACKED"),
    (24, 2880, 2880, False, "ACCUMULATORS"),
    (16, 1024, 6144, False, None),
    (28, 2048, 3072, True, "PACKED"),
    (28, 1024, 6144, True, None),
    (32, 2880, 2112, True, "ACCUMULATORS"),
    (33, 1536, 4096, True, "PACKED"),
    (48, 1536, 4096, False, "PACKED"),
    (33, 4096, 3072, True, "ACCUMULATORS"),
    (64, 1024, 6144, True, "PACKED"),
    (64, 1536, 4096, True, "ACCUMULATORS"),
    (64, 4096, 3072, True, "ACCUMULATORS"),
    (64, 4096, 4096, False, "ACCUMULATORS"),
    (33, 1024, 4096, False, None),
    (33, 1024, 6208, True, None),
    (65, 4096, 4096, False, None),
    # small weights keep the kernels above
    (8, 1024, 1024, False, None),
    (13, 1024, 2048, True, None),
    (16, 2048, 2048, True, None),
]
# The narrow tile without full stages on sm_100a: two corners of each window, the
# measured shapes just past a window's bound that keep the other kernels, and a
# shape inside a window whose input is not a multiple of 64 (no narrow tile).
SM100_NARROW_ROUTES = [
    (1, 32, 8192, True, "PACKED_NARROW"),
    (1, 32, 10240, True, "PACKED_NARROW"),
    (1, 32, 11264, True, "PACKED_NARROW"),
    (1, 256, 4096, True, None),
    (1, 512, 9216, True, None),
    (2, 32, 7168, True, None),
    (2, 128, 4096, True, None),
    (2, 128, 12296, True, None),
    (2, 256, 3584, True, "PACKED_NARROW"),
    (3, 32, 3584, True, "PACKED_NARROW"),
    (3, 32, 7168, True, "PACKED_NARROW"),
    (3, 64, 5120, True, None),
    (3, 128, 3072, True, None),
    (3, 128, 5120, True, None),
    (3, 384, 9216, True, None),
    (4, 32, 3584, True, "PACKED_NARROW"),
    (4, 64, 3072, True, None),
    (4, 128, 6144, True, None),
    (4, 256, 3072, True, None),
    (4, 320, 5120, True, None),
    (4, 320, 7168, True, None),
    (4, 320, 20480, True, "PACKED_NARROW"),
    (4, 512, 20480, True, "PACKED_NARROW"),
    (6, 32, 6144, True, "PACKED_NARROW"),
    (6, 256, 20480, True, "PACKED_NARROW"),
    (6, 320, 4096, True, "PACKED_NARROW"),
    (6, 320, 9216, True, None),
    (6, 384, 4096, True, None),
    (6, 384, 10240, True, None),
    (6, 384, 20480, True, "PACKED_NARROW"),
    (6, 448, 20480, True, None),
    (6, 512, 15360, True, None),
    (7, 32, 7168, True, "PACKED_NARROW"),
    (7, 128, 7168, True, "PACKED_NARROW"),
    (8, 64, 20480, True, "PACKED_NARROW"),
    (8, 128, 6144, True, "PACKED_NARROW"),
    (8, 128, 7168, True, "PACKED_NARROW"),
    (8, 128, 8192, True, "PACKED_NARROW"),
    (8, 128, 20480, True, "PACKED_NARROW"),
    (8, 160, 7168, True, "PACKED_NARROW"),
    (8, 256, 4096, True, "PACKED_NARROW"),
    (8, 256, 9216, True, None),
    (8, 320, 4096, True, None),
    (8, 320, 18432, True, None),
    (9, 128, 7168, True, "PACKED_NARROW"),
    (9, 192, 9216, True, "PACKED_NARROW"),
    (9, 256, 8192, True, None),
    (1, 32, 7168, False, "PACKED_NARROW"),
    (1, 32, 8192, False, "PACKED_NARROW"),
    (1, 128, 6144, False, None),
    (2, 32, 5120, False, None),
    (2, 32, 6144, False, "PACKED_NARROW"),
    (2, 128, 3584, False, None),
    (2, 320, 6144, False, None),
    (2, 512, 7168, False, None),
    (3, 32, 3584, False, "PACKED_NARROW"),
    (3, 32, 5120, False, "PACKED_NARROW"),
    (3, 32, 6144, False, "PACKED_NARROW"),
    (3, 256, 3584, False, None),
    (4, 32, 4096, False, None),
    (4, 128, 4096, False, None),
    (4, 128, 5120, False, None),
    (4, 512, 20480, False, "PACKED_NARROW"),
    (6, 32, 3072, False, None),
    (6, 64, 3584, False, "PACKED_NARROW"),
    (6, 256, 5120, False, None),
    (6, 320, 20480, False, "PACKED_NARROW"),
    (6, 384, 6144, False, None),
    (6, 384, 7168, False, None),
    (6, 384, 20480, False, "PACKED_NARROW"),
    (6, 448, 10240, False, None),
    (6, 512, 15360, False, None),
    (8, 32, 4096, False, None),
    (8, 64, 3072, False, None),
    (8, 64, 20480, False, "PACKED_NARROW"),
    (8, 128, 3584, False, "PACKED_NARROW"),
    (8, 128, 20480, False, "PACKED_NARROW"),
    (8, 256, 20480, False, "PACKED_NARROW"),
    (8, 320, 9216, False, None),
    (8, 384, 14336, False, None),
]
# sm_100a, rows without a bias from 3584 output features: PACKED_BOX8 on 1 to 7 rows
# at 4480 to 4608 output features and inputs of 2048 to 8192, and at 3584 to 4608
# from an input of 6152; at 8192 output features PACKED_M64_N8 on inputs that are a
# multiple of 64 and PACKED_BOX8 on the others; PACKED_M64_N8 in its three other
# windows (corners and inside points).  The shapes just past each edge, off a
# multiple of 64 where a window needs one, on 9 rows or with a bias go to the next
# window or keep their kernels; the other 8-row shapes outside the 8192 and 12288
# windows keep their kernels or go to the last windows, listed below.
SM100_BOX8_M64_N8_ROUTES = [
    (1, 4480, 2048, False, "PACKED_BOX8"),
    (7, 4608, 8192, False, "PACKED_BOX8"),
    (3, 4544, 5000, False, "PACKED_BOX8"),
    (4, 4480, 2040, False, None),
    (4, 4608, 8256, False, None),
    (1, 4464, 4096, False, "PACKED_BOX8"),
    (7, 4624, 4096, False, "PACKED_M64_N8"),
    (1, 3584, 6152, False, "PACKED_BOX8"),
    (5, 3584, 8192, False, "PACKED_BOX8"),
    (6, 4096, 7432, False, "PACKED_BOX8"),
    (3, 4096, 6144, False, "PACKED"),
    (3, 3568, 7168, False, "PACKED_M64_N8"),
    (3, 3584, 8256, False, None),
    (8, 4096, 7168, False, "PACKED_BOX8"),
    (8, 4608, 8192, False, "PACKED_BOX8"),
    (8, 8192, 2048, False, "PACKED_M64_N8"),
    (8, 8192, 8192, False, "PACKED_M64_N8"),
    (1, 8192, 2112, False, "PACKED_M64_N8"),
    (5, 8192, 5632, False, "PACKED_M64_N8"),
    (8, 8192, 2056, False, "PACKED_BOX8"),
    (1, 8192, 8136, False, "PACKED_BOX8"),
    (4, 8192, 5000, False, "PACKED_BOX8"),
    (8, 8192, 2040, False, None),
    (8, 8192, 8200, False, None),
    (8, 8192, 8256, False, None),
    (7, 8208, 4104, False, None),
    (8, 8192, 4096, True, None),
    (1, 4624, 2048, False, "PACKED_M64_N8"),
    (7, 8176, 6152, False, "PACKED_M64_N8"),
    (3, 6656, 4104, False, "PACKED_M64_N8"),
    (5, 6992, 6160, False, None),
    (9, 6656, 4096, False, "PACKED"),
    (3, 6656, 4096, True, None),
    (1, 7008, 8192, False, "PACKED_M64_N8"),
    (6, 7680, 7168, False, "PACKED_M64_N8"),
    (3, 7680, 7176, False, None),
    (2, 7680, 8256, False, None),
    (5, 6992, 7168, False, None),
    (8, 4480, 2048, False, "PACKED_M64_N8"),
    (8, 4608, 6152, False, "PACKED_BOX8"),
    (8, 10000, 3000, False, None),
    (8, 8208, 4104, False, None),
    (8, 16384, 6152, False, None),
    (8, 7008, 8192, False, None),
    (8, 16384, 8184, False, None),
    (8, 12288, 7000, False, None),
    (8, 12288, 2048, False, "PACKED_M64_N8"),
    (8, 12288, 6144, False, "PACKED_M64_N8"),
    (8, 12288, 10240, False, "PACKED_M64_N8"),
    (8, 12288, 16384, False, "PACKED_M64_N8"),
    (8, 12288, 2040, False, None),
    (8, 12288, 16376, False, None),
    (8, 12288, 16448, False, None),
    (8, 12272, 12288, False, None),
    (8, 12304, 12288, False, None),
    (7, 12288, 12288, False, None),
    (8, 12288, 12288, True, None),
]
# sm_100a, the last windows, for rows without a bias that no window above admits:
# PACKED_M64_N8 on 8 rows at inputs that are a multiple of 64 (4480 to 7167 output
# features at inputs of 2048 to 4352 and 5632 to 6144, 7168 to 16384 at 2048 to 6144,
# 7168 to 9216 above 6152 up to 8192), PACKED on 9 to 16 rows from 4224 output
# features, PACKED_BOX8 at 4112 to 4464, and PACKED_M64_N8, PACKED_BOX8 and PACKED in
# the windows past those.  Corners, inside points and inputs that are not a multiple of
# 64; the shapes just past an edge, on other row counts or with a bias go to another
# window or keep their kernels.
SM100_LAST_WINDOWS_ROUTES = [
    # 8 rows at 4480 to 7167 output features, inputs of 2048 to 4352 and 5632 to 6144
    (8, 4480, 2112, False, "PACKED_M64_N8"),
    (8, 7152, 4352, False, "PACKED_M64_N8"),
    (8, 5120, 2560, False, "PACKED_M64_N8"),
    (8, 4480, 5632, False, "PACKED_M64_N8"),
    (8, 7152, 6144, False, "PACKED_M64_N8"),
    (8, 6144, 5888, False, "PACKED_M64_N8"),
    (8, 4464, 2048, False, "PACKED_BOX8"),
    (8, 6144, 4416, False, None),
    (8, 6144, 5568, False, None),
    (8, 6144, 2040, False, None),
    (8, 6144, 4104, False, None),
    (8, 6144, 6208, False, None),
    (9, 6144, 4096, False, "PACKED"),
    (8, 6144, 4096, True, None),
    # 8 rows at 7168 to 16384 output features and inputs of 2048 to 6144
    (8, 7168, 2048, False, "PACKED_M64_N8"),
    (8, 16384, 6144, False, "PACKED_M64_N8"),
    (8, 10240, 3072, False, "PACKED_M64_N8"),
    (8, 14336, 4096, False, "PACKED_M64_N8"),
    (8, 7168, 4104, False, None),
    (8, 16384, 6208, False, None),
    (7, 10240, 4096, False, "PACKED_M64_N8"),
    (8, 9216, 4096, True, None),
    # 8 rows at 7168 to 9216 output features and inputs above 6152 up to 8192
    (8, 7168, 6208, False, "PACKED_M64_N8"),
    (8, 9216, 8192, False, "PACKED_M64_N8"),
    (8, 7664, 7168, False, "PACKED_M64_N8"),
    (8, 7680, 6208, False, "PACKED_M64_N8"),
    (8, 8704, 7680, False, "PACKED_M64_N8"),
    (8, 9232, 7168, False, None),
    (8, 8704, 8256, False, None),
    (8, 8704, 7176, False, None),
    (8, 7152, 7168, False, None),
    (7, 8704, 7168, False, None),
    (8, 8704, 7168, True, None),
    # PACKED on 9 to 16 rows at 4224 to 16384 output features
    (9, 4224, 2048, False, "PACKED"),
    (16, 16384, 12288, False, "PACKED"),
    (12, 8192, 6152, False, "PACKED"),
    (16, 10240, 3080, False, "PACKED"),
    (16, 16400, 4096, False, None),
    (9, 6144, 12296, False, None),
    (9, 6144, 2040, False, None),
    (17, 6144, 4096, False, None),
    (12, 8192, 6152, True, None),
    # PACKED_BOX8 at 4112 to 4464 output features and inputs of 2048 to 6144
    (8, 4112, 2048, False, "PACKED_BOX8"),
    (8, 4464, 6144, False, "PACKED_BOX8"),
    (8, 4288, 4104, False, "PACKED_BOX8"),
    (1, 4112, 2048, False, "PACKED_BOX8"),
    (7, 4464, 6144, False, "PACKED_BOX8"),
    (4, 4288, 4096, False, "PACKED_BOX8"),
    (4, 4288, 4104, False, None),
    (8, 4096, 5120, False, "PACKED"),
    (9, 4288, 4096, False, "PACKED"),
    (8, 4288, 4104, True, None),
    # 1 to 7 rows from 9216 output features, and at 8192 from an input of 10240
    (1, 9216, 2048, False, "PACKED_M64_N8"),
    (7, 16384, 4096, False, "PACKED_M64_N8"),
    (4, 12288, 3072, False, "PACKED_M64_N8"),
    (7, 16400, 4096, False, None),
    (4, 12288, 4160, False, None),
    (4, 12288, 3080, False, None),
    (4, 12288, 2040, False, None),
    (4, 12288, 3072, True, None),
    (1, 8192, 10240, False, "PACKED_M64_N8"),
    (7, 8192, 12288, False, "PACKED_M64_N8"),
    (4, 8192, 11264, False, "PACKED_M64_N8"),
    (4, 8192, 10176, False, None),
    (4, 8192, 12352, False, None),
    (4, 8192, 11272, False, None),
    (4, 8208, 11264, False, None),
    (8, 8192, 11264, False, None),
    (4, 8192, 11264, True, None),
    # 8 rows at 16400 to 24576 output features and inputs of 2048 to 6144
    (8, 16400, 2048, False, "PACKED_M64_N8"),
    (8, 24576, 6144, False, "PACKED_M64_N8"),
    (8, 20480, 4096, False, "PACKED_M64_N8"),
    (8, 24592, 4096, False, None),
    (8, 20480, 6208, False, None),
    (8, 20480, 4104, False, None),
    (8, 20480, 1984, False, None),
    (7, 20480, 4096, False, None),
    (8, 20480, 4096, True, None),
    # 8 rows at 6144 to 16384 output features and inputs of 1024 to 1984
    (8, 6144, 1024, False, "PACKED_M64_N8"),
    (8, 16384, 1984, False, "PACKED_M64_N8"),
    (8, 10240, 1536, False, "PACKED_M64_N8"),
    (8, 6128, 1536, False, None),
    (8, 16400, 1536, False, None),
    (8, 10240, 1544, False, None),
    (8, 10240, 960, False, None),
    (8, 10240, 2048, False, "PACKED_M64_N8"),
    (7, 10240, 1536, False, None),
    (8, 10240, 1536, True, None),
    # 1 to 8 rows at 2880 to 3568 output features and inputs of 7168 to 16384
    (1, 2880, 7168, False, "PACKED_M64_N8"),
    (8, 3568, 16384, False, "PACKED_M64_N8"),
    (4, 3072, 12288, False, "PACKED_M64_N8"),
    (4, 2864, 12288, False, None),
    (4, 3584, 12288, False, None),
    (4, 3072, 7104, False, None),
    (4, 3072, 16448, False, None),
    (4, 3072, 12296, False, None),
    (9, 3072, 12288, False, None),
    (4, 3072, 12288, True, None),
    # 8 rows at 7168 to 7808 output features from an input of 8200
    (8, 7168, 8256, False, "PACKED_M64_N8"),
    (8, 7808, 12288, False, "PACKED_M64_N8"),
    (8, 7424, 10240, False, "PACKED_M64_N8"),
    (8, 7168, 8200, False, "PACKED_BOX8"),
    (8, 7808, 9992, False, "PACKED_BOX8"),
    (8, 7424, 9000, False, "PACKED_BOX8"),
    (8, 7808, 10000, False, None),
    (8, 7824, 10240, False, None),
    (8, 7152, 10240, False, None),
    (8, 7424, 12352, False, None),
    (7, 7424, 10240, False, None),
    (8, 7424, 10240, True, None),
    # 8 rows at 10240 to 16384 output features, inputs not a multiple of 64
    (8, 10240, 2056, False, "PACKED_M64_N8"),
    (8, 16384, 6136, False, "PACKED_M64_N8"),
    (8, 14336, 4104, False, "PACKED_M64_N8"),
    (8, 12288, 4104, False, "PACKED_M64_N8"),
    (8, 10224, 4104, False, None),
    (8, 16400, 4104, False, None),
    (8, 14336, 6152, False, None),
    (8, 14336, 2040, False, None),
    (8, 14336, 4104, True, None),
    # 8 rows at 4480 to 4608 output features and inputs of 2048 to 8192 that the
    # PACKED_M64_N8 windows above do not take
    (8, 4480, 2056, False, "PACKED_BOX8"),
    (8, 4608, 2056, False, "PACKED_BOX8"),
    (8, 4480, 8192, False, "PACKED_BOX8"),
    (8, 4544, 4800, False, "PACKED_BOX8"),
    (8, 4544, 4104, False, "PACKED_BOX8"),
    (8, 4496, 7688, False, "PACKED_BOX8"),
    (8, 4592, 6144, False, "PACKED_M64_N8"),
    (8, 4624, 7168, False, None),
    (8, 4480, 8200, False, None),
    (8, 4480, 2040, False, None),
    (9, 4544, 4800, False, "PACKED"),
    (7, 4544, 4800, False, "PACKED_BOX8"),
    (8, 4544, 4800, True, None),
    # 8 rows at 3584 to 4479 output features and inputs of 6152 to 8192
    (8, 3584, 6152, False, "PACKED_BOX8"),
    (8, 4464, 8192, False, "PACKED_BOX8"),
    (8, 3584, 8192, False, "PACKED_BOX8"),
    (8, 4464, 6152, False, "PACKED_BOX8"),
    (8, 3840, 7176, False, "PACKED_BOX8"),
    (8, 4224, 6400, False, "PACKED_BOX8"),
    (8, 3568, 7176, False, None),
    (8, 4096, 8200, False, None),
    (8, 4096, 6144, False, "PACKED"),
    (9, 4096, 7168, False, None),
    (7, 4096, 7168, False, "PACKED_BOX8"),
    (8, 4096, 7168, True, None),
    # 1 to 7 rows at 8193 to 9215 output features and inputs of 2048 to 4096 that are
    # a multiple of 64
    (1, 8208, 2048, False, "PACKED_M64_N8"),
    (7, 9200, 4096, False, "PACKED_M64_N8"),
    (1, 9200, 2048, False, "PACKED_M64_N8"),
    (7, 8208, 4096, False, "PACKED_M64_N8"),
    (4, 8704, 3072, False, "PACKED_M64_N8"),
    (4, 8704, 3080, False, None),
    (4, 8704, 4160, False, None),
    (4, 8704, 2040, False, None),
    (8, 8704, 3072, False, "PACKED_M64_N8"),
    (4, 8704, 3072, True, None),
    # PACKED on 9 to 16 rows at 4097 to 4223 output features and inputs of 2048 to 12288
    (9, 4112, 2048, False, "PACKED"),
    (16, 4208, 12288, False, "PACKED"),
    (16, 4112, 12288, False, "PACKED"),
    (9, 4208, 4096, False, "PACKED"),
    (12, 4160, 7176, False, "PACKED"),
    (9, 4096, 8192, False, None),
    (9, 4160, 12296, False, None),
    (16, 4160, 2040, False, None),
    (17, 4160, 4096, False, None),
    (8, 4160, 4096, False, "PACKED_BOX8"),
    (12, 4160, 4096, True, None),
]
# sm_103a, rows without a bias from 3584 output features: PACKED_BOX8 on 8 rows at
# 4480 to 4608 output features and inputs of 2048 to 12288; at 8192 output features
# PACKED_M64_N8 on inputs that are a multiple of 64 and PACKED_BOX8 on the others;
# PACKED_M64_N8 in its five other windows (corners and inside points).  The shapes
# just past each edge, off a multiple of 64 where a window needs one, on 9 rows or
# with a bias go to another window or keep their kernels.  1 to 7 rows at 4480 to
# 4608 output features, and at 3584 to 4608 from an input of 6152, keep the
# reference, as do 8 rows at 3584 to 4464 output features and inputs of 6152 to
# 12288.
SM103_BOX8_M64_N8_ROUTES = [
    (8, 4480, 2048, False, "PACKED_BOX8"),
    (8, 4608, 12288, False, "PACKED_BOX8"),
    (8, 4544, 7000, False, "PACKED_BOX8"),
    (8, 4464, 4096, False, "PACKED_BOX8"),
    (8, 4624, 4096, False, "PACKED_M64_N8"),
    (8, 4480, 2040, False, "PACKED_M64_N8"),
    (8, 4608, 12296, False, None),
    (9, 4480, 4096, False, "PACKED"),
    (8, 4480, 4096, True, None),
    (1, 4480, 2048, False, None),
    (7, 4608, 8192, False, None),
    (3, 4544, 5000, False, None),
    (4, 4480, 2040, False, None),
    (4, 4608, 8256, False, None),
    (1, 4464, 4096, False, None),
    (7, 4624, 4096, False, "PACKED_M64_N8"),
    (1, 4608, 2048, False, None),
    (7, 4608, 6152, False, None),
    (4, 4624, 6152, False, "PACKED_M64_N8"),
    (8, 3584, 6152, False, None),
    (8, 3584, 12288, False, None),
    (8, 4096, 12288, False, None),
    (8, 4096, 9000, False, None),
    (8, 4096, 6144, False, "PACKED"),
    (8, 3568, 8192, False, "PACKED_M64_N8"),
    (8, 4096, 12296, False, None),
    (1, 3584, 6152, False, None),
    (7, 4096, 8192, False, None),
    (5, 4000, 7432, False, None),
    (3, 4096, 6144, False, "PACKED"),
    (3, 3568, 7168, False, "PACKED_M64_N8"),
    (3, 3584, 8256, False, None),
    (8, 8192, 2048, False, "PACKED_M64_N8"),
    (8, 8192, 8192, False, "PACKED_M64_N8"),
    (1, 8192, 2112, False, "PACKED_M64_N8"),
    (5, 8192, 5632, False, "PACKED_M64_N8"),
    (8, 8192, 2056, False, "PACKED_BOX8"),
    (8, 8192, 8136, False, "PACKED_BOX8"),
    (1, 8192, 8136, False, "PACKED_BOX8"),
    (4, 8192, 5000, False, "PACKED_BOX8"),
    (8, 8192, 2040, False, "PACKED_M64_N8"),
    (3, 8192, 2040, False, None),
    (8, 8192, 8200, False, None),
    (7, 8192, 8256, False, None),
    (7, 8208, 4104, False, None),
    (8, 8192, 4096, True, None),
    (8, 4624, 2048, False, "PACKED_M64_N8"),
    (8, 16384, 6144, False, "PACKED_M64_N8"),
    (8, 10000, 4096, False, "PACKED_M64_N8"),
    (8, 16384, 6152, False, None),
    (8, 10000, 3000, False, "PACKED_M64_N8"),
    (8, 16400, 4096, False, "PACKED_M64_N8"),
    (8, 10000, 6208, False, None),
    (8, 4624, 2040, False, "PACKED_M64_N8"),
    (9, 6656, 4096, False, "PACKED"),
    (8, 6656, 4096, True, None),
    (1, 4624, 2048, False, "PACKED_M64_N8"),
    (7, 8176, 6152, False, "PACKED_M64_N8"),
    (3, 6656, 4104, False, "PACKED_M64_N8"),
    (5, 6992, 6160, False, None),
    (2, 8208, 4096, False, "PACKED_M64_N8"),
    (3, 6656, 2040, False, None),
    (8, 8192, 8256, False, "PACKED_M64_N8"),
    (8, 8192, 10240, False, "PACKED_M64_N8"),
    (8, 8192, 12288, False, "PACKED_M64_N8"),
    (8, 8192, 12280, False, None),
    (8, 8192, 12352, False, None),
    (8, 8208, 10240, False, None),
    (7, 8192, 10240, False, None),
    (8, 12288, 2048, False, "PACKED_M64_N8"),
    (8, 12288, 6208, False, "PACKED_M64_N8"),
    (8, 12288, 12288, False, "PACKED_M64_N8"),
    (8, 12288, 12280, False, None),
    (8, 12288, 12352, False, None),
    (8, 12272, 8192, False, None),
    (8, 12304, 8192, False, None),
    (7, 12288, 8192, False, "PACKED_M64_N8"),
    (8, 12288, 8192, True, None),
    (8, 3584, 8192, False, None),
    (8, 3584, 12352, False, "PACKED_M64_N8"),
    (8, 3584, 14336, False, "PACKED_M64_N8"),
    (8, 3584, 16384, False, "PACKED_M64_N8"),
    (8, 3584, 16376, False, None),
    (8, 3584, 16448, False, None),
    (8, 3600, 14336, False, None),
    (8, 3568, 14336, False, None),
    (7, 3584, 14336, False, None),
    (8, 3584, 14336, True, None),
]
# sm_103a, the last windows, for rows without a bias that no window above admits:
# PACKED_M64_N8 on 8 rows at 7424 to 9216 output features and inputs of 6153 to 8192,
# on 1 to 7 rows at 7424 to 8192 output features and those inputs when they are a
# multiple of 64, and on 8 rows at 5008 to 7423 output features and inputs of 2048 to
# 5631 and at 7424 to 16384 output features and inputs of 2048 to 6144; PACKED on 9 to
# 16 rows at 4097 to 16384 output features and inputs of 2048 to 12288; PACKED_BOX8 on
# 8 rows at 4097 to 4479 output features and inputs of 2048 to 6144; PACKED_M64_N8 on 1
# to 7 rows above 8192 output features, on 8 rows above 16384 output features or at
# inputs of 1024 to 2047, and on 1 to 8 rows at 2880 to 3583 output features.  Corners,
# inside points and inputs that are not a multiple of 64; the shapes just past an
# edge, on other row counts or with a bias go to another window or keep their kernels.
SM103_LAST_WINDOWS_ROUTES = [
    # 8 rows at 7424 to 9216 output features (8192 goes to the windows above)
    (8, 7424, 6160, False, "PACKED_M64_N8"),
    (8, 8176, 8192, False, "PACKED_M64_N8"),
    (8, 7808, 7176, False, "PACKED_M64_N8"),
    (8, 8208, 6160, False, "PACKED_M64_N8"),
    (8, 9216, 8184, False, "PACKED_M64_N8"),
    (8, 8704, 7176, False, "PACKED_M64_N8"),
    (8, 7408, 7168, False, None),
    (8, 7424, 6152, False, None),
    (8, 7424, 8200, False, None),
    (8, 9232, 7176, False, None),
    (8, 8192, 7176, False, "PACKED_BOX8"),
    (8, 7424, 6160, True, None),
    (9, 7424, 6160, False, "PACKED"),
    (7, 9216, 8192, False, "PACKED_M64_N8"),
    # 1 to 7 rows at 7424 to 8192 output features, inputs a multiple of 64
    (1, 7424, 6208, False, "PACKED_M64_N8"),
    (7, 8176, 8192, False, "PACKED_M64_N8"),
    (4, 7680, 7168, False, "PACKED_M64_N8"),
    (6, 7424, 8128, False, "PACKED_M64_N8"),
    (1, 7408, 6208, False, None),
    (6, 7680, 7176, False, None),
    (3, 7680, 8256, False, None),
    (3, 7680, 6208, True, None),
    (5, 8192, 6208, False, "PACKED_M64_N8"),
    # 8 rows at 5008 to 7423 output features and inputs of 2048 to 5631
    (8, 5008, 2056, False, "PACKED_M64_N8"),
    (8, 7408, 5624, False, "PACKED_M64_N8"),
    (8, 6144, 5128, False, "PACKED_M64_N8"),
    (8, 5632, 4104, False, "PACKED_M64_N8"),
    (8, 4992, 4104, False, None),
    (8, 6144, 5640, False, None),
    (8, 6656, 6136, False, None),
    (8, 6144, 2040, False, "PACKED_M64_N8"),
    (8, 6144, 5128, True, None),
    (9, 6144, 5128, False, "PACKED"),
    # 8 rows at 7424 to 16384 output features and inputs of 2048 to 6144
    (8, 7424, 2056, False, "PACKED_M64_N8"),
    (8, 16384, 6136, False, "PACKED_M64_N8"),
    (8, 12288, 3080, False, "PACKED_M64_N8"),
    (8, 7424, 6136, False, "PACKED_M64_N8"),
    (8, 16400, 6136, False, "PACKED_M64_N8"),
    (8, 7424, 6136, True, None),
    # PACKED on 9 to 16 rows at 4097 to 16384 output features
    (9, 4112, 2048, False, "PACKED"),
    (16, 16384, 6144, False, "PACKED"),
    (12, 7168, 3080, False, "PACKED"),
    (9, 4112, 6152, False, "PACKED"),
    (16, 16384, 12288, False, "PACKED"),
    (10, 5120, 8200, False, "PACKED"),
    (11, 4608, 12280, False, "PACKED"),
    (17, 6144, 4096, False, None),
    (9, 4112, 2040, False, None),
    (16, 16400, 4096, False, None),
    (9, 6144, 12296, False, None),
    (16, 16400, 12288, False, None),
    (12, 7168, 3072, True, None),
    # PACKED_BOX8 on 8 rows at 4097 to 4479 output features
    (8, 4112, 2048, False, "PACKED_BOX8"),
    (8, 4464, 6144, False, "PACKED_BOX8"),
    (8, 4288, 4104, False, "PACKED_BOX8"),
    (8, 4464, 6152, False, None),
    (8, 4480, 4096, False, "PACKED_BOX8"),
    (8, 4112, 2040, False, "PACKED_M64_N8"),
    (7, 4288, 4104, False, None),
    (8, 4288, 4104, True, None),
    # 1 to 7 rows above 8192 output features
    (1, 8208, 2048, False, "PACKED_M64_N8"),
    (7, 16384, 4096, False, "PACKED_M64_N8"),
    (4, 12288, 3080, False, "PACKED_M64_N8"),
    (1, 16400, 2048, False, None),
    (3, 12288, 2040, False, None),
    (7, 16384, 4104, False, None),
    (4, 12288, 3080, True, None),
    (1, 8208, 4160, False, "PACKED_M64_N8"),
    (4, 10240, 6144, False, "PACKED_M64_N8"),
    (2, 9216, 7176, False, None),
    (3, 12288, 6152, False, None),
    (1, 12304, 8192, False, None),
    (6, 12288, 8256, False, None),
    # 8 rows above 16384 output features
    (8, 16400, 2048, False, "PACKED_M64_N8"),
    (8, 24576, 6144, False, "PACKED_M64_N8"),
    (8, 20480, 4104, False, "PACKED_M64_N8"),
    (8, 24592, 4096, False, None),
    (8, 20480, 6152, False, None),
    (8, 20480, 2040, False, None),
    (7, 20480, 4096, False, None),
    (8, 20480, 4096, True, None),
    # 8 rows at inputs of 1024 to 2047
    (8, 3584, 1024, False, "PACKED_M64_N8"),
    (8, 16384, 2040, False, "PACKED_M64_N8"),
    (8, 8192, 1544, False, "PACKED_M64_N8"),
    (8, 3584, 1984, False, "PACKED"),
    (8, 3568, 1536, False, None),
    (8, 6144, 1016, False, None),
    (8, 16400, 2040, False, None),
    (7, 8192, 1544, False, None),
    (8, 8192, 1544, True, None),
    # 1 to 8 rows at 2880 to 3583 output features, inputs a multiple of 64
    (1, 2880, 6208, False, "PACKED_M64_N8"),
    (8, 3568, 12288, False, "PACKED_M64_N8"),
    (4, 3072, 8192, False, "PACKED_M64_N8"),
    (2, 3200, 7176, False, None),
    (8, 2864, 8192, False, None),
    (1, 3568, 12352, False, None),
    (9, 3072, 8192, False, None),
    (4, 3072, 8192, True, None),
    (8, 3568, 6144, False, "PACKED"),
]
# The full-stage narrow tile on sm_103a: two corners of each window, the measured
# shapes just past a window's bound that keep the other kernels, and a shape inside
# a window whose input is not a multiple of 64 (no narrow tile).
SM103_NARROW_ROUTES = [
    (1, 32, 3584, True, "PACKED_NARROW_FULL_STAGES"),
    (1, 32, 8192, True, "PACKED_NARROW_FULL_STAGES"),
    (1, 320, 3072, True, None),
    (1, 512, 7168, True, None),
    (2, 64, 3072, True, None),
    (2, 128, 4096, True, None),
    (2, 128, 7168, True, None),
    (2, 256, 2560, True, None),
    (2, 448, 3584, True, None),
    (2, 512, 20480, True, "PACKED_NARROW_FULL_STAGES"),
    (3, 32, 3072, True, "PACKED_NARROW_FULL_STAGES"),
    (3, 32, 7168, True, "PACKED_NARROW_FULL_STAGES"),
    (3, 128, 6144, True, "PACKED_NARROW_FULL_STAGES"),
    (3, 256, 2560, True, "PACKED_NARROW_FULL_STAGES"),
    (3, 256, 5120, True, None),
    (3, 384, 3072, True, None),
    (3, 512, 11264, True, None),
    (4, 32, 8192, True, "PACKED_NARROW_FULL_STAGES"),
    (4, 64, 5120, True, None),
    (4, 64, 6144, True, None),
    (4, 128, 2560, True, None),
    (4, 320, 5120, True, None),
    (4, 448, 7168, True, None),
    (4, 512, 20480, True, "PACKED_NARROW_FULL_STAGES"),
    (6, 320, 2048, True, None),
    (6, 288, 3400, True, None),
    (6, 320, 4096, True, "PACKED_NARROW_FULL_STAGES"),
    (6, 384, 3584, True, "PACKED_NARROW_FULL_STAGES"),
    (6, 384, 20480, True, "PACKED_NARROW_FULL_STAGES"),
    (6, 448, 9216, True, None),
    (6, 512, 15360, True, None),
    (8, 64, 2560, True, None),
    (8, 256, 4096, True, "PACKED_NARROW_FULL_STAGES"),
    (8, 256, 20480, True, "PACKED_NARROW_FULL_STAGES"),
    (8, 320, 3584, True, None),
    (8, 320, 4096, True, None),
    (8, 320, 18432, True, None),
    (1, 32, 7168, False, "PACKED_NARROW_FULL_STAGES"),
    (1, 32, 8192, False, "PACKED_NARROW_FULL_STAGES"),
    (1, 256, 6144, False, None),
    (1, 320, 5120, False, None),
    (1, 320, 6144, False, "PACKED_NARROW_FULL_STAGES"),
    (1, 384, 6144, False, None),
    (1, 384, 8192, False, "PACKED_NARROW_FULL_STAGES"),
    (2, 32, 6144, False, "PACKED_NARROW_FULL_STAGES"),
    (2, 64, 3584, False, None),
    (2, 512, 20480, False, "PACKED_NARROW_FULL_STAGES"),
    (3, 32, 3584, False, "PACKED_NARROW_FULL_STAGES"),
    (3, 32, 4096, False, None),
    (3, 256, 5120, False, None),
    (3, 448, 7168, False, None),
    (4, 32, 3584, False, "PACKED_NARROW_FULL_STAGES"),
    (4, 64, 3584, False, "PACKED_NARROW_FULL_STAGES"),
    (4, 64, 5120, False, "PACKED_NARROW_FULL_STAGES"),
    (4, 128, 4096, False, "PACKED_NARROW_FULL_STAGES"),
    (4, 160, 3072, False, None),
    (4, 512, 20480, False, "PACKED_NARROW_FULL_STAGES"),
    (5, 352, 7168, False, None),
    (6, 32, 3072, False, None),
    (6, 64, 20480, False, "PACKED_NARROW_FULL_STAGES"),
    (6, 320, 7168, False, "PACKED_NARROW_FULL_STAGES"),
    (6, 320, 20480, False, "PACKED_NARROW_FULL_STAGES"),
    (6, 384, 7168, False, None),
    (6, 384, 20480, False, "PACKED_NARROW_FULL_STAGES"),
    (6, 512, 15360, False, None),
    (7, 128, 4096, False, "PACKED_NARROW_FULL_STAGES"),
    (8, 32, 4096, False, None),
    (8, 32, 5120, False, None),
    (8, 192, 4608, False, "PACKED_NARROW_FULL_STAGES"),
    (8, 192, 5120, False, "PACKED_NARROW_FULL_STAGES"),
    (8, 256, 3584, False, "PACKED_NARROW_FULL_STAGES"),
    (8, 256, 4096, False, None),
    (8, 256, 20480, False, "PACKED_NARROW_FULL_STAGES"),
    (8, 320, 3584, False, None),
    (8, 320, 9216, False, None),
    (8, 384, 14336, False, None),
]
# sm_107a: the narrow tile up to 8 rows over the measured windows (inside points,
# the nearest measured shapes the windows exclude and an input that is not a
# multiple of 64), PACKED on 8 rows without a bias and 9 to 16 with one from 3584
# output features, with a bias 28 to 32 rows from 2048 output features, without one
# 9 to 16 rows from 1728; PACKED_BOX8 on 8 rows without a bias from 3584 output
# features and an input of 3072 to 6144 (the corners; below 3072 PACKED keeps the
# rows; fewer output features, 7 or 9 rows, a longer input and a bias keep their
# kernels or go to the later windows below); PACKED_M64_N8 on 8 rows without a bias
# ahead of it, at an input of 2048 from 3584 output features and at inputs that are
# a multiple of 64 in its three other windows (corners, odd multiples of 64; the
# same shapes off a multiple of 64, past each edge, 7 or 9 rows and a bias keep
# their kernels or go to the later windows below), and PACKED_BOX8 at
# 4096 to 5120 output features and inputs of 2112 to 2560 (corners, an input that
# is not a multiple of 64, past each edge); the M=64 tiles at 2048 to 3072 output
# features (corners,
# and the measured shapes past them); PACKED on 17 to 23 rows without a bias whose
# activations take less than 128 KiB, from an input of 2048 on 17 to 21 rows and from
# 2816 / 2688 on 22 / 23 rows (the corners, the first input past 128 KiB, an input
# below the start, 24 rows and a bias keep their kernels).
SM107_ROUTES = [
    (4, 384, 16384, True, "PACKED_NARROW_U4"),
    (3, 512, 16384, True, "PACKED_NARROW_U4"),
    (4, 448, 16384, True, "PACKED_NARROW_U4"),
    (5, 256, 3072, True, "PACKED_NARROW_U4"),
    (7, 320, 16384, True, "PACKED_NARROW_U4"),
    (3, 512, 4096, True, "PACKED_NARROW_U4"),
    (3, 512, 16384, False, "PACKED_NARROW_U4"),
    (3, 512, 16392, False, None),
    (4, 384, 16384, False, "PACKED_NARROW_U4"),
    (5, 384, 16384, False, "PACKED_NARROW_U4"),
    (2, 512, 16384, False, "PACKED_NARROW_U4"),
    (5, 128, 16384, False, "PACKED_NARROW_U4"),
    (1, 32, 11264, True, None),
    (8, 448, 11264, True, None),
    (1, 256, 9216, True, None),
    (8, 448, 18432, True, None),
    (2, 256, 8192, True, None),
    (4, 32, 9216, True, None),
    (6, 512, 9216, True, None),
    (6, 384, 7168, True, None),
    (2, 64, 3072, True, None),
    (8, 32, 3072, True, None),
    (8, 256, 2048, True, None),
    (3, 128, 5120, True, None),
    (4, 128, 7168, True, None),
    (8, 64, 7168, True, None),
    (6, 128, 6144, True, None),
    (8, 512, 4096, True, None),
    (1, 32, 4096, True, None),
    (8, 448, 15360, False, None),
    (1, 32, 7168, False, None),
    (1, 256, 5120, False, None),
    (2, 32, 7168, False, None),
    (3, 32, 6144, False, None),
    (6, 512, 6144, False, None),
    (2, 64, 5120, False, None),
    (3, 32, 5120, False, None),
    (6, 256, 5120, False, "PACKED_NARROW_U4"),
    (6, 128, 4096, False, "PACKED_NARROW_U4"),
    (4, 64, 4096, False, "PACKED_NARROW_U4"),
    (8, 320, 8192, False, "PACKED_NARROW_U4"),
    (4, 160, 3072, False, None),
    (4, 2880, 2880, False, "PACKED_M64_N4"),
    (4, 2048, 10240, False, "PACKED_M64_N4"),
    (4, 1536, 4096, False, None),
    (4, 2048, 2880, True, "PACKED_M64_N4"),
    (4, 3072, 8192, True, "PACKED_M64_N4"),
    (3, 2048, 8192, True, None),
    (4, 3584, 2880, True, None),
    (8, 4096, 4096, False, "PACKED_M64_N8"),
    (8, 3584, 3072, False, "PACKED_BOX8"),
    (8, 4096, 6144, False, "PACKED_M64_N8"),
    (8, 3856, 4600, False, "PACKED_BOX8"),
    (8, 4096, 3064, False, "PACKED"),
    (8, 3584, 2112, False, "PACKED"),
    (8, 3856, 3064, False, "PACKED"),
    (8, 4096, 2560, False, "PACKED_BOX8"),
    (8, 3584, 2048, False, "PACKED_M64_N8"),
    (8, 8192, 2048, False, "PACKED_M64_N8"),
    (8, 5120, 2048, False, "PACKED_M64_N8"),
    (8, 3568, 2048, False, None),
    (8, 8208, 2048, False, None),
    (8, 3584, 2040, False, None),
    (8, 4096, 2056, False, "PACKED_BOX8"),
    (7, 8192, 2048, False, None),
    (9, 8192, 2048, False, "PACKED"),
    (8, 4096, 2048, True, None),
    (8, 7168, 2112, False, "PACKED_M64_N8"),
    (8, 7168, 8192, False, "PACKED_M64_N8"),
    (8, 8192, 4160, False, "PACKED_M64_N8"),
    (8, 7168, 2056, False, None),
    (8, 7152, 2112, False, "PACKED_M64_N8"),
    (8, 8192, 8256, False, None),
    (8, 8208, 4096, False, None),
    (8, 6144, 3072, False, "PACKED_M64_N8"),
    (8, 6144, 8192, False, "PACKED_M64_N8"),
    (8, 6144, 3136, False, "PACKED_M64_N8"),
    (8, 6144, 3080, False, None),
    (8, 6144, 3008, False, "PACKED_M64_N8"),
    (8, 6128, 4096, False, "PACKED_M64_N8"),
    (8, 6144, 4096, True, None),
    (8, 3584, 4096, False, "PACKED_M64_N8"),
    (8, 4608, 8192, False, "PACKED_M64_N8"),
    (8, 4096, 4160, False, "PACKED_M64_N8"),
    (8, 4608, 6208, False, "PACKED_M64_N8"),
    (8, 4096, 4104, False, "PACKED_BOX8"),
    (8, 3584, 4032, False, "PACKED_BOX8"),
    (8, 4608, 4104, False, "PACKED_BOX8"),
    (8, 4624, 4096, False, "PACKED_M64_N8"),
    (8, 4096, 8256, False, None),
    (7, 4096, 4160, False, None),
    (9, 4608, 4160, False, "PACKED"),
    (8, 4096, 2112, False, "PACKED_BOX8"),
    (8, 5120, 2560, False, "PACKED_BOX8"),
    (8, 4608, 2120, False, "PACKED_BOX8"),
    (8, 4096, 2568, False, "PACKED"),
    (8, 5136, 2560, False, None),
    (8, 5120, 2104, False, None),
    (8, 5120, 2568, False, None),
    (8, 4096, 2112, True, None),
    (8, 3568, 4096, False, None),
    (8, 4096, 6152, False, "PACKED_BOX8"),
    (7, 4096, 4096, False, None),
    (9, 4096, 4096, False, "PACKED"),
    (8, 4096, 4096, True, None),
    (9, 2880, 2112, True, None),
    (16, 2880, 2880, True, "PACKED_M64_N16"),
    (16, 3072, 10240, True, "PACKED_M64_N16"),
    (11, 3072, 6144, True, "PACKED_M64_N16"),
    (9, 2048, 8192, True, "PACKED_M64_N16"),
    (16, 2560, 2048, True, None),
    (17, 2880, 4096, True, None),
    (16, 2880, 2880, False, "PACKED_M64_N16"),
    (16, 2048, 2048, False, "PACKED_M64_N16"),
    (9, 3072, 10240, False, "PACKED_M64_N16"),
    (12, 2880, 4096, False, "PACKED"),
    (12, 1600, 4096, False, None),
    (16, 1664, 6144, False, None),
    (9, 1728, 4096, False, "PACKED"),
    (17, 1536, 4096, False, "PACKED"),
    (28, 1536, 4096, False, "PACKED"),
    (28, 1536, 4096, True, None),
    (32, 2048, 3072, True, "PACKED"),
    (64, 4096, 3072, True, "ACCUMULATORS"),
    (17, 4096, 3848, False, "PACKED"),
    (17, 4096, 3856, False, "ACCUMULATORS"),
    (22, 3408, 2976, False, "PACKED"),
    (21, 3456, 2048, False, "PACKED"),
    (22, 3408, 2816, False, "PACKED"),
    (22, 3408, 2808, False, "ACCUMULATORS"),
    (23, 3456, 2688, False, "PACKED"),
    (23, 3456, 2680, False, "ACCUMULATORS"),
    (23, 3456, 2048, False, "ACCUMULATORS"),
    (23, 4096, 2856, False, "ACCUMULATORS"),
    (20, 4096, 2040, False, "ACCUMULATORS"),
    (24, 4096, 2048, False, "ACCUMULATORS"),
    (20, 4096, 3272, True, None),
]
# sm_107a, rows without a bias after PACKED's 8-row window: PACKED_M64_N8 on 8 rows
# at 4097 to 6784 output features and inputs from 2561 that are a multiple of 64 (up
# to 8192 below 6144 output features, up to 3071 from there), PACKED_BOX8 on 8 rows
# at 3584 to 6784 output features and the other inputs of 2056 to 8184,
# PACKED_M64_N8 on 4 and 5 rows at 3584 to 4608 output features and an input of
# 2048, PACKED_M64_N4 on 4 rows at 4097 to 6144 output features and inputs of 3072
# to 6144 that are a multiple of 64 (corners and inside points).  The tests
# count 148 SMs, where PACKED_BOX8's one-wave windows end at 4736 output features.
# The shapes just past each edge, off a multiple of 64 where a window needs one, on
# other rows or with a bias go to another window or keep their kernels.
SM107_BOX8_M64_ROUTES = [
    (8, 4112, 2624, False, "PACKED_M64_N8"),
    (8, 6128, 8192, False, "PACKED_M64_N8"),
    (8, 5120, 5120, False, "PACKED_M64_N8"),
    (8, 4608, 4032, False, "PACKED_M64_N8"),
    (8, 6128, 8256, False, None),
    (8, 4096, 3008, False, "PACKED"),
    (8, 6144, 4096, False, "PACKED_M64_N8"),
    (9, 5120, 5120, False, "PACKED"),
    (8, 5120, 5120, True, None),
    (8, 6144, 2624, False, "PACKED_M64_N8"),
    (8, 6784, 3008, False, "PACKED_M64_N8"),
    (8, 6400, 2816, False, "PACKED_M64_N8"),
    (8, 6800, 2624, False, None),
    (8, 6784, 3072, False, "PACKED_M64_N8"),
    (8, 3584, 2056, False, "PACKED_BOX8"),
    (8, 4736, 2104, False, "PACKED_BOX8"),
    (8, 4000, 2056, False, "PACKED_BOX8"),
    (8, 4752, 2104, False, None),
    (8, 6784, 2560, False, None),
    (8, 3584, 6152, False, "PACKED_BOX8"),
    (8, 4096, 8184, False, "PACKED_BOX8"),
    (8, 4736, 2568, False, "PACKED_BOX8"),
    (8, 4608, 5000, False, "PACKED_BOX8"),
    (8, 4752, 2568, False, None),
    (8, 6784, 8184, False, None),
    (8, 4096, 8200, False, None),
    (8, 3568, 7000, False, None),
    (9, 4608, 5000, False, "PACKED"),
    (8, 4608, 5000, True, None),
    (4, 4112, 3072, False, "PACKED_M64_N4"),
    (4, 6144, 6144, False, "PACKED_M64_N4"),
    (4, 5120, 4096, False, "PACKED_M64_N4"),
    (4, 4096, 4096, False, None),
    (4, 6160, 4096, False, None),
    (4, 5120, 3008, False, None),
    (4, 5120, 6208, False, None),
    (4, 5120, 4104, False, None),
    (3, 5120, 4096, False, None),
    (5, 5120, 4096, False, None),
    (4, 5120, 4096, True, None),
    (4, 3584, 2048, False, "PACKED_M64_N8"),
    (5, 4608, 2048, False, "PACKED_M64_N8"),
    (4, 4096, 2048, False, "PACKED_M64_N8"),
    (4, 4624, 2048, False, None),
    (5, 3568, 2048, False, None),
    (4, 4096, 2056, False, None),
    (3, 4096, 2048, False, None),
    (4, 4096, 2048, True, None),
]
# sm_107a, the last windows, for rows without a bias that no window above admits:
# PACKED_M64_N8 on 8 rows above 6784 and up to 7167 output features at inputs of 2112
# to 2304 that are a multiple of 64, on 6 and 7 rows at 4096 output features and an
# input of 2048, and on 8 rows above 6784 and up to 7056 output features at inputs of
# 7169 to 8184 that are not a multiple of 64; then PACKED on 9 to 16 rows at 4097 to
# 16384 output features and inputs of 1024 to 16384, except above 6784 output features
# at inputs above 12288, where PACKED_M64_N16 serves them.  Corners, inside points,
# the edges between windows and inputs that are not a multiple of 64; the shapes just
# past an edge, at inputs a window does not take, on other row counts or with a bias
# go to another window or keep their kernels.
SM107_LAST_WINDOWS_ROUTES = [
    # 8 rows, 6785 to 7167 output features, inputs of 2112 to 2304
    (8, 6800, 2112, False, "PACKED_M64_N8"),
    (8, 7152, 2304, False, "PACKED_M64_N8"),
    (8, 6976, 2176, False, "PACKED_M64_N8"),
    (8, 6784, 2112, False, None),
    (8, 7168, 2304, False, "PACKED_M64_N8"),
    (8, 6800, 2048, False, "PACKED_M64_N8"),
    (8, 7152, 2368, False, None),
    (8, 6976, 2120, False, None),
    (7, 6976, 2176, False, None),
    (9, 6976, 2176, False, "PACKED"),
    (8, 6976, 2176, True, None),
    # 6 and 7 rows, 4096 output features, an input of 2048
    (6, 4096, 2048, False, "PACKED_M64_N8"),
    (7, 4096, 2048, False, "PACKED_M64_N8"),
    (7, 4080, 2048, False, None),
    (6, 4112, 2048, False, None),
    (7, 4096, 2056, False, None),
    (6, 4096, 1984, False, None),
    (6, 4096, 2560, False, None),
    (7, 4096, 3136, False, None),
    (5, 4096, 2048, False, "PACKED_M64_N8"),
    (8, 4096, 2048, False, "PACKED_M64_N8"),
    (6, 4096, 2048, True, None),
    # 8 rows, 6785 to 7056 output features, inputs of 7169 to 8184
    (8, 6800, 7176, False, "PACKED_M64_N8"),
    (8, 7056, 8184, False, "PACKED_M64_N8"),
    (8, 6912, 7432, False, "PACKED_M64_N8"),
    (8, 6800, 7680, False, "PACKED_M64_N8"),
    (8, 7072, 7176, False, None),
    (8, 6784, 7176, False, None),
    (8, 6800, 7160, False, None),
    (8, 6912, 8200, False, None),
    (7, 6912, 7432, False, None),
    (9, 6912, 7432, False, "PACKED"),
    (8, 6912, 7432, True, None),
    # 9 to 16 rows, 4097 to 16384 output features, inputs of 1024 to 16384
    (9, 4112, 1024, False, "PACKED"),
    (16, 6784, 16384, False, "PACKED"),
    (9, 6784, 16384, False, "PACKED"),
    (16, 4112, 1024, False, "PACKED"),
    (12, 5120, 7176, False, "PACKED"),
    (13, 6144, 12296, False, "PACKED"),
    (9, 6800, 1024, False, "PACKED"),
    (16, 16384, 12288, False, "PACKED"),
    (16, 6800, 12288, False, "PACKED"),
    (9, 16384, 1024, False, "PACKED"),
    (11, 12288, 4104, False, "PACKED"),
    (14, 8192, 8192, False, "PACKED"),
    (10, 9216, 12280, False, "PACKED"),
    (9, 6800, 12296, False, "PACKED_M64_N16"),
    (16, 16384, 16384, False, "PACKED_M64_N16"),
    (16, 6800, 16384, False, "PACKED_M64_N16"),
    (9, 16384, 12296, False, "PACKED_M64_N16"),
    (10, 10240, 14344, False, "PACKED_M64_N16"),
    (15, 7168, 16384, False, "PACKED_M64_N16"),
    (12, 6784, 12296, False, "PACKED"),
    (12, 8192, 12288, False, "PACKED"),
    (12, 4096, 4096, False, "PACKED"),
    (12, 4096, 12288, False, None),
    (12, 16400, 8192, False, None),
    (12, 16400, 16384, False, None),
    (12, 6784, 1016, False, None),
    (12, 12288, 1016, False, None),
    (12, 6784, 16392, False, None),
    (12, 12288, 16392, False, None),
    (8, 8192, 8192, False, "PACKED_M64_N8"),
    (17, 8192, 8192, False, None),
    (17, 12288, 16384, False, None),
    (12, 8192, 8192, True, None),
    (12, 12288, 16384, True, None),
]
ROUTES = (
    [
        (capability, *route)
        for capability in ((10, 0), (10, 3))
        for route in SM100_FAMILY_ROUTES
    ]
    + [((10, 0), *route) for route in SM100_NARROW_ROUTES]
    + [((10, 0), *route) for route in SM100_BOX8_M64_N8_ROUTES]
    + [((10, 0), *route) for route in SM100_LAST_WINDOWS_ROUTES]
    + [((10, 3), *route) for route in SM103_NARROW_ROUTES]
    + [((10, 3), *route) for route in SM103_BOX8_M64_N8_ROUTES]
    + [((10, 3), *route) for route in SM103_LAST_WINDOWS_ROUTES]
    + [((10, 7), *route) for route in SM107_ROUTES]
    + [((10, 7), *route) for route in SM107_BOX8_M64_ROUTES]
    + [((10, 7), *route) for route in SM107_LAST_WINDOWS_ROUTES]
    + [((9, 0), 64, 4096, 3072, True, None)]
)


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
    monkeypatch.setattr(routergemm, "_multiprocessor_count", lambda _: 148)
    monkeypatch.delenv("FLASHINFER_DISABLE_TINYGEMM2_TCGEN05", raising=False)
    input = torch.empty(batch_size, input_features, dtype=torch.bfloat16)
    weight = torch.empty(output_features, input_features, dtype=torch.bfloat16)
    out = torch.empty(batch_size, output_features, dtype=torch.bfloat16)
    program = routergemm._tinygemm2_tcgen05_program(
        input, weight, out, has_bias, use_pdl
    )
    expected = (
        None
        if tile is None
        else tile + ("" if has_bias else "_NOBIAS") + ("_PDL" if use_pdl else "")
    )
    assert program == expected


def test_tinygemm2_tcgen05_disabled_and_unaligned(monkeypatch):
    """The escape hatch, a tensor-map-unaligned operand and, for ACCUMULATORS, an
    output that is not 32-byte aligned keep the kernels the call runs otherwise."""
    import flashinfer.gemm.routergemm as routergemm

    monkeypatch.setattr(routergemm, "get_compute_capability", lambda _: (10, 0))
    monkeypatch.setattr(routergemm, "_multiprocessor_count", lambda _: 148)
    input = torch.empty(64, 4096, dtype=torch.bfloat16)
    weight = torch.empty(4096, 4096, dtype=torch.bfloat16)
    out = torch.empty(64, 4096, dtype=torch.bfloat16)
    assert (
        routergemm._tinygemm2_tcgen05_program(input, weight, out, True, False)
        == "ACCUMULATORS"
    )
    monkeypatch.setenv("FLASHINFER_DISABLE_TINYGEMM2_TCGEN05", "1")
    assert (
        routergemm._tinygemm2_tcgen05_program(input, weight, out, True, False) is None
    )
    monkeypatch.delenv("FLASHINFER_DISABLE_TINYGEMM2_TCGEN05")
    storage = torch.empty(64 * 4096 + 1, dtype=torch.bfloat16)
    shifted = storage[1:].view(64, 4096)
    assert (
        routergemm._tinygemm2_tcgen05_program(shifted, weight, out, True, False) is None
    )
    assert (
        routergemm._tinygemm2_tcgen05_program(
            torch.empty(64, 4100, dtype=torch.bfloat16),
            torch.empty(4096, 4100, dtype=torch.bfloat16),
            out,
            True,
            False,
        )
        is None
    )
    out_storage = torch.empty(64 * 4096 + 16, dtype=torch.bfloat16)
    assert out_storage.data_ptr() % 32 == 0
    aligned = out_storage[16:].view(64, 4096)
    unaligned = out_storage[8 : 8 + 64 * 4096].view(64, 4096)
    assert (
        routergemm._tinygemm2_tcgen05_program(input, weight, aligned, True, False)
        == "ACCUMULATORS"
    )
    assert (
        routergemm._tinygemm2_tcgen05_program(input, weight, unaligned, True, False)
        is None
    )


@pytest.mark.parametrize("has_bias", [True, False])
def test_tinygemm_bf16_unaligned_out_bitwise(has_bias):
    """An output that is not 32-byte aligned on a shape ACCUMULATORS serves gets
    the reference's bits from the kernel the call runs otherwise."""
    _target()
    from flashinfer.gemm.routergemm import tinygemm_bf16

    input, weight, bias = _make_case(64, 4096, 4096)
    if not has_bias:
        bias = None
    out_ref = _reference(input, weight, bias, False)
    storage = torch.full(
        (64 * 4096 + 8,), float("nan"), device="cuda", dtype=torch.bfloat16
    )
    out = storage[8:].view(64, 4096)
    assert out.data_ptr() % 32 != 0
    tinygemm_bf16(input, weight, out, bias=bias)
    torch.cuda.synchronize()
    assert torch.equal(out, out_ref)


@pytest.mark.parametrize("program", ["ACCUMULATORS", "ACCUMULATORS_NOBIAS_PDL"])
def test_tinygemm2_tcgen05_op_rejects_unaligned_out(program):
    """Called directly, an ACCUMULATORS program refuses an output that is not
    32-byte aligned and launches nothing."""
    _target()
    from flashinfer.gemm.routergemm import get_tinygemm2_tcgen05_op

    input, weight, bias = _make_case(64, 4096, 4096)
    if "_NOBIAS" in program:
        bias = None
    storage = torch.full(
        (64 * 4096 + 8,), float("nan"), device="cuda", dtype=torch.bfloat16
    )
    out = storage[8:].view(64, 4096)
    assert out.data_ptr() % 32 != 0
    with pytest.raises(ValueError, match="32-byte aligned"):
        get_tinygemm2_tcgen05_op()(program, input, weight, bias, out)
    torch.cuda.synchronize()
    assert torch.isnan(storage).all()


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
