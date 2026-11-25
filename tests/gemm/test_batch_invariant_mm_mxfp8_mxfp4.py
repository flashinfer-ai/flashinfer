# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Mixed MXFP8-by-MXFP4 GEMM: dequantized references, row invariance, PDL and CUDA Graph replay."""

import dataclasses
import inspect

import pytest
import torch

import flashinfer
from flashinfer import batch_invariant as bi
from flashinfer.gemm.batch_invariant.api import (
    mm_mxfp8_mxfp4_batch_invariant,
    part_of,
)
from flashinfer.gemm.batch_invariant.blocks.operands import can_implement
from flashinfer.gemm.batch_invariant.generator import (
    REFERENCE_SHAPE,
    Knobs,
    enumerate_space,
    knob_product,
)
from flashinfer.gemm.batch_invariant.heuristic import PARTS, hbm_traffic
from flashinfer.gemm.batch_invariant.schedule import (
    chunks_for,
    default_config,
    mx_class_for,
    rank_topk,
)
from flashinfer.gemm.batch_invariant.spec import (
    MX_MIXED_FMTS,
    chunk_unit,
    k_granule,
    spec_fingerprint,
)

E4 = torch.float8_e4m3fn
needs_device = pytest.mark.usefixtures("cudnn_batch_invariant_device")
#: The e2m1 codes' values (the low nibble holds element 2j, the high nibble 2j + 1).
E2M1_VALUES = torch.tensor(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
    dtype=torch.float64,
)


def bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16) if t.element_size() == 2 else t.view(torch.int32)


def swizzle_sf(sf_rows: torch.Tensor) -> torch.Tensor:
    """The (rows, K / 32) uint8 block scales in the 128x4 swizzled layout the quantizers write: atoms of 128 rows x 4 blocks
    (512 bytes; row r, block j of an atom at byte (r % 32) * 16 + (r // 32) * 4 + j), K-atoms consecutive, row-atoms strided
    by the K-atoms of the whole K; the padding bytes are zero."""
    rows, kb = sf_rows.shape
    rt, kt = -(-rows // 128), -(-kb // 4)
    buf = torch.zeros(rt * 128, kt * 4, dtype=torch.uint8)
    buf[:rows, :kb] = sf_rows.cpu()
    r = torch.arange(rt * 128).view(-1, 1)
    j = torch.arange(kt * 4).view(1, -1)
    off = (
        (r // 128) * (kt * 512)
        + (j // 4) * 512
        + (r % 32) * 16
        + ((r % 128) // 32) * 4
        + (j % 4)
    )
    flat = torch.empty(rt * 128 * kt * 4, dtype=torch.uint8)
    flat[off.reshape(-1)] = buf.reshape(-1)
    return flat.cuda()


def e8m0(sf_flat: torch.Tensor, rows: int, K: int) -> torch.Tensor:
    """The (rows, K / 32) scale values (2^(byte - 127)) of a swizzled buffer, in fp64."""
    kb = K // 32
    rt, kt = -(-rows // 128), -(-kb // 4)
    r = torch.arange(rt * 128).view(-1, 1)
    j = torch.arange(kt * 4).view(1, -1)
    off = (
        (r // 128) * (kt * 512)
        + (j // 4) * 512
        + (r % 32) * 16
        + ((r % 128) // 32) * 4
        + (j % 4)
    )
    b = sf_flat.cpu()[off.reshape(-1)].view(rt * 128, kt * 4)[:rows, :kb].double()
    return torch.exp2(b - 127)


def sf_bytes_of(sf_flat: torch.Tensor, rows: int, K: int) -> torch.Tensor:
    """The (rows, K / 32) UE8M0 bytes of a swizzled buffer (to re-swizzle a row subset)."""
    return (torch.log2(e8m0(sf_flat, rows, K)) + 127).to(torch.uint8)


def adv_mx8(rows: int, K: int, seed: int, cancel: bool = True, span: int = 12):
    """An adversarial MXFP8 activation: e4m3 values over the format's range with random signs, near-cancelling neighbour
    pairs and zeros of both signs; UE8M0 block scales spread over 2^-span .. 2^span so that the products of neighbouring
    blocks differ by orders of magnitude and any accumulation-order difference reaches the output bits."""
    g = torch.Generator().manual_seed(seed)
    e = (torch.rand(rows, K, generator=g) * 2 - 1) * (7 if span >= 12 else 3)
    x = (torch.rand(rows, K, generator=g) * 2 - 1) * torch.exp2(e)
    if cancel and K >= 8:
        k = torch.arange(0, K - 1, 8)
        x[:, k + 1] = -x[:, k] * (1 + 2.0**-3)
    z = torch.randint(0, 61, (rows, K), generator=g)
    x[z == 0] = 0.0
    x[z == 1] = -0.0
    sf = torch.randint(127 - span, 127 + span + 1, (rows, K // 32), generator=g).to(
        torch.uint8
    )
    return x.to(E4).cuda(), swizzle_sf(sf)


def adv_mx4(rows: int, K: int, seed: int, span: int = 12):
    """An adversarial MXFP4 weight: every e2m1 code with equal frequency (both zeros included), packed two per byte, with
    UE8M0 block scales over 2^-span .. 2^span. Returns the packed (rows, K / 2) uint8 tensor and its scales."""
    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 16, (rows, K), generator=g)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
    sf = torch.randint(127 - span, 127 + span + 1, (rows, K // 32), generator=g).to(
        torch.uint8
    )
    return packed.cuda(), swizzle_sf(sf)


def weight(N: int, K: int, seed: int, span: int = 12):
    """The entry's weight: the (K / 2, N) transpose of a row-major (N, K / 2) packed e2m1 tensor and its scales."""
    w, sf = adv_mx4(N, K, seed, span=span)
    return w.t(), sf


def unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """The values of a packed (rows, K / 2) e2m1 tensor as an fp64 (rows, K) tensor."""
    p = packed.cpu().long()
    out = torch.empty(p.shape[0], p.shape[1] * 2, dtype=torch.float64)
    out[:, 0::2] = E2M1_VALUES[p & 0xF]
    out[:, 1::2] = E2M1_VALUES[p >> 4]
    return out


def dequant8(q: torch.Tensor, sf_flat: torch.Tensor) -> torch.Tensor:
    rows, K = q.shape
    return q.double().cpu() * e8m0(sf_flat, rows, K).repeat_interleave(32, dim=1)


def dequant4(packed: torch.Tensor, sf_flat: torch.Tensor) -> torch.Tensor:
    rows, K = packed.shape[0], packed.shape[1] * 2
    return unpack_e2m1(packed) * e8m0(sf_flat, rows, K).repeat_interleave(32, dim=1)


def norm_rel_err(c, a, sfa, bt, sfb, alpha: float = 1.0) -> float:
    """The max norm-relative error against the exact product of the dequantised operands (times alpha)."""
    A, B = dequant8(a, sfa), dequant4(bt, sfb)
    ref = (A @ B.t()) * alpha
    scale = (A.abs() @ B.abs().t()) * abs(alpha)
    return ((c.double().cpu() - ref).abs() / (scale + 1e-30)).max().item()


def mm(a, b, sfa, sfb, alpha=None, out_dtype=torch.bfloat16, **kw):
    return flashinfer.mm_mxfp8_mxfp4(
        a, b, sfa, sfb, alpha, out_dtype=out_dtype, batch_invariant=True, **kw
    )


def alpha_t(v: float) -> torch.Tensor:
    return torch.full((1,), float(v), device="cuda", dtype=torch.float32)


_ORACLES = {}


def fi_sm107_mixed(
    a, bt, sfa, sfb, alpha, out_dtype=torch.bfloat16, tile=(128, 128), cluster=(1, 1)
):
    """FlashInfer's ``Sm107BlockScaledPersistentDenseGemmKernel`` on its mixed path (``A{e4m3} x B{e2m1}``: the same
    ``kind::mxf8f6f4`` K = 64 instruction, the packed e2m1 operand, two steps per stage, K ascending, alpha then one
    conversion), driven through its pointer entry (the tensor entry rejects mixed dtypes): the bit-equality oracle of
    ``gemm.mxfp4w8a.kfold64``. That kernel rounds alpha to the output type, so a power-of-two alpha for bf16 / f16 outputs."""
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_ptr

    from flashinfer.cute_dsl.utils import get_max_active_clusters
    from flashinfer.gemm.kernels.dense_blockscaled_gemm_sm107 import (
        Sm107BlockScaledPersistentDenseGemmKernel,
    )

    M, K = a.shape
    N = bt.shape[0]
    assert bt.shape[1] * 2 == K
    sf_m, sf_n, sf_k = -(-M // 128), -(-N // 128), -(-(K // 32) // 4)
    c_dt = {
        torch.bfloat16: cutlass.BFloat16,
        torch.float16: cutlass.Float16,
        torch.float32: cutlass.Float32,
    }[out_dtype]
    key = (tile, cluster, out_dtype)
    if key not in _ORACLES:
        gemm = Sm107BlockScaledPersistentDenseGemmKernel(
            32,
            (tile[0], tile[1], 64),
            (tile[0], tile[1], 128),
            cluster,
            prefetch_dist=0,
        )
        mac = get_max_active_clusters(cluster[0] * cluster[1])
        a_ptr = make_ptr(
            cutlass.Float8E4M3FN, 16, cute.AddressSpace.gmem, assumed_align=16
        )
        b_ptr = make_ptr(
            cutlass.Float4E2M1FN, 16, cute.AddressSpace.gmem, assumed_align=16
        )
        sf_ptr = make_ptr(
            cutlass.Float8E8M0FNU, 16, cute.AddressSpace.gmem, assumed_align=16
        )
        c_ptr = make_ptr(c_dt, 16, cute.AddressSpace.gmem, assumed_align=16)
        alpha_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Float32, (1,), assumed_align=4
        )
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        _ORACLES[key] = cute.compile(
            gemm.wrapper_ptrs,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            a_ptr,
            b_ptr,
            sf_ptr,
            sf_ptr,
            c_ptr,
            alpha_fake,
            0,
            mac,
            stream_fake,
            False,
            options="--opt-level 2 --enable-tvm-ffi",
        )
    out = torch.empty(M, N, dtype=out_dtype, device=a.device)
    a, bt = a.contiguous(), bt.contiguous()
    _ORACLES[key](
        M,
        N,
        K,
        sf_m,
        sf_n,
        sf_k,
        a.data_ptr(),
        bt.data_ptr(),
        sfa.data_ptr(),
        sfb.data_ptr(),
        out.data_ptr(),
        alpha.reshape(1).float().contiguous(),
        0,
    )
    return out


# ----------------------------------------------------------------------------------------------- device-free


def test_fingerprint_is_a_function_of_the_weight_shape_the_output_format_and_the_architecture():
    f107 = bi.fingerprint(
        "mm_mxfp8_mxfp4", N=4096, K=4096, out_dtype=torch.bfloat16, arch="sm_107"
    )
    f100 = bi.fingerprint(
        "mm_mxfp8_mxfp4", N=4096, K=4096, out_dtype=torch.bfloat16, arch="sm_100"
    )
    assert f107.startswith("gemm-mxfp4w8a-k64-kfold-scaled-") and f100.startswith(
        "gemm-mxfp4w8a-kfold-scaled-"
    )
    assert f107 != f100  # two instructions, two classes
    assert f100 == bi.fingerprint(
        "mm_mxfp8_mxfp4", N=4096, K=4096, out_dtype="bf16", arch="sm_103"
    )
    assert f107 == bi.fingerprint(
        "mm_mxfp8_mxfp4", N=4096, K=4096, out_dtype="bf16", arch=(10, 7)
    )
    assert bi.fingerprint(
        "mm_mxfp8_mxfp4", N=128, K=8192, out_dtype=torch.float16, arch="sm_100"
    ).startswith("gemm-mxfp4w8a-kchunk4-scaled-")
    # a weight in the mixed classes' lower-bound region carries the split class (8 chunks) on both parts; the MX rule would give 2
    assert bi.fingerprint(
        "mm_mxfp8_mxfp4", N=512, K=16384, out_dtype=torch.bfloat16, arch="sm_100"
    ).startswith("gemm-mxfp4w8a-kchunk8-scaled-")
    assert bi.fingerprint(
        "mm_mxfp8_mxfp4", N=512, K=16384, out_dtype=torch.bfloat16, arch="sm_107"
    ).startswith("gemm-mxfp4w8a-k64-kchunk8-scaled-")
    assert bi.fingerprint(
        "mm_mxfp8", N=512, K=16384, out_dtype=torch.bfloat16, arch="sm_100"
    ).startswith("gemm-mxfp8-kchunk2-")
    assert bi.fingerprint(
        "mm_mxfp8_mxfp4", N=1024, K=16384, out_dtype=torch.bfloat16, arch="sm_100"
    ).startswith("gemm-mxfp4w8a-kfold-scaled-")
    # the output dtype and the raw K are dimensions of the specification; M and the layout of the scale buffers are not
    distinct = {
        bi.fingerprint("mm_mxfp8_mxfp4", N=4096, K=K, out_dtype=od, arch=arch)
        for K in (4096, 4096 + 128)
        for od in (torch.bfloat16, torch.float16)
        for arch in ("sm_100", "sm_107")
    }
    assert len(distinct) == 8
    # a different class from the one-format MX classes of the same shape and part: the step's operand formats differ
    for arch in ("sm_100", "sm_107"):
        mixed = bi.fingerprint(
            "mm_mxfp8_mxfp4", N=4096, K=4096, out_dtype=torch.bfloat16, arch=arch
        )
        assert mixed != bi.fingerprint(
            "mm_mxfp8", N=4096, K=4096, out_dtype=torch.bfloat16, arch=arch
        )
        assert mixed != bi.fingerprint(
            "mm_fp4",
            N=4096,
            K=4096,
            out_dtype=torch.bfloat16,
            use_nvfp4=False,
            arch=arch,
        )
    assert spec_fingerprint(4096, "bf16", 1, "mxfp4w8a") == f100
    assert spec_fingerprint(4096, "bf16", 1, "mxfp4w8a_k64") == f107
    assert bi.class_name("mm_mxfp8_mxfp4") == "gemm.mxfp4w8a"
    with pytest.raises(TypeError):
        bi.fingerprint("mm_mxfp8_mxfp4", N=4096, K=4096)
    with pytest.raises(ValueError):
        bi.fingerprint(
            "mm_mxfp8_mxfp4",
            N=4096,
            K=4096,
            out_dtype=torch.float32,
            arch="sm_100",
        )
    with pytest.raises(ValueError):
        bi.fingerprint(
            "mm_mxfp8_mxfp4", N=4096, K=4096, out_dtype=torch.bfloat16, arch="sm_90"
        )


def test_chunk_rule_and_granules():
    """One chunk rule for every MX class (128-element granules: a whole scale-factor atom column and whole steps of either
    instruction); the body's K granule is 128 for the SM100 class (the TMA's unpacking runs over the e2m1 row) and 32 for the
    SM107 one (one block scale, 16-byte rows of the packed weight)."""
    for fmt in MX_MIXED_FMTS:
        assert chunk_unit(fmt) == 128
        for N, K in [(4096, 4096), (128, 4096), (64, 8192), (128, 8192), (64, 16640)]:
            assert chunks_for(N, K, fmt) == chunks_for(N, K, "mxfp8")
        # the mixed classes' own lower bound: 8 chunks for N <= 512, K >= 16384 in whole runs of 8 granules; the MX rule elsewhere
        assert [
            chunks_for(N, K, fmt)
            for N, K in (
                (256, 16384),
                (512, 16384),
                (512, 32768),
                (384, 65536),
                (128, 16384),
            )
        ] == [8, 8, 8, 8, 8]
        assert [
            chunks_for(N, K, fmt)
            for N, K in (
                (1024, 16384),
                (520, 16384),
                (512, 8192),
                (512, 15360),
                (512, 16384 + 384),
                (64, 65536),
            )
        ] == [1, 1, 1, 1, 1, 64]
        for N, K in (
            (1024, 16384),
            (520, 16384),
            (512, 8192),
            (2048, 2048),
            (64, 65536),
        ):
            assert chunks_for(N, K, fmt) == chunks_for(N, K, "mxfp8")
    assert chunks_for(512, 16384, "mxfp8") == 2 and chunks_for(512, 16384, "mxfp4") == 2
    assert k_granule("mxfp4w8a") == 128 and k_granule("mxfp4w8a_k64") == 32
    assert (
        k_granule("mxfp8") == 32
        and k_granule("mxfp4") == 32
        and k_granule("nvfp4") == 32
    )
    assert (
        mx_class_for("mxfp4w8a", "sm_107") == "mxfp4w8a_k64"
        and mx_class_for("mxfp4w8a", "sm_100") == "mxfp4w8a"
        and mx_class_for("mxfp4w8a", "sm_103") == "mxfp4w8a"
    )


def test_block_scaled_spaces_and_acceptance():
    """The mixed classes' knob products are the MXFP8 classes' (the format is a class, not a knob: 424 / 520 records, the
    B-reuse tile on the SM107 class only); the kernel takes K-major weights only; the SM100 class accepts whole 128-element
    runs of K only, the SM107 class the MX granule; the instruction strings name the operand formats, exchanged by the
    swapped records."""
    assert (
        len(knob_product(in_fmt="mxfp4w8a")) == 424
        and len(knob_product(in_fmt="mxfp4w8a_k64")) == 520
    )
    for fmt, twin in (("mxfp4w8a", "mxfp8"), ("mxfp4w8a_k64", "mxfp8_k64")):
        prod = knob_product(in_fmt=fmt)
        # the records are the MXFP8 twin's with the k-block erased: the SM107 class's swapped records stage 256 elements
        # (`Knobs.k_block`, the weight's TMA row at 128 bytes), every other record the 128-byte rule
        assert [dataclasses.replace(k, k_block=0) for k in prod] == knob_product(
            in_fmt=twin
        )
        kb = 256 if fmt == "mxfp4w8a_k64" else 0
        assert all(k.k_block == (kb if k.swap_ab else 0) for k in prod)
        assert all(
            k.describe().endswith("_swap_kb256") == (k.swap_ab and kb == 256)
            for k in prod
        )
        assert enumerate_space(*REFERENCE_SHAPE, b_major="n", in_fmt=fmt) == []
        ref = enumerate_space(*REFERENCE_SHAPE, b_major="k", in_fmt=fmt)
        assert ref == enumerate_space(*REFERENCE_SHAPE, b_major="k", in_fmt=twin)
        dec = enumerate_space(8, 4096, 4096, b_major="k", in_fmt=fmt)
        assert set(ref) < set(dec) and all(k.swap_ab for k in set(dec) - set(ref))
        c4 = enumerate_space(16, 128, 8192, chunks=4, b_major="k", in_fmt=fmt)
        assert any(k.cluster_k == 4 for k in c4) and any(k.cta_chunks for k in c4)
        # the split classes of the mixed formats take the swapped records (the workspace split only); the MXFP8 twin's do not
        sw4 = [k for k in c4 if k.swap_ab]
        assert sw4 and all(
            k.cluster_k == 1 and not k.cta_chunks and k.tile_n >= 16 for k in sw4
        )
        assert not any(
            k.swap_ab
            for k in enumerate_space(16, 128, 8192, chunks=4, b_major="k", in_fmt=twin)
        )
        assert [
            dataclasses.replace(k, k_block=0) for k in c4 if not k.swap_ab
        ] == enumerate_space(16, 128, 8192, chunks=4, b_major="k", in_fmt=twin)
        dense = Knobs((128, 128), (1, 1), False, True, "m", 1)
        for K in (128, 256, 3968, 4096, 65536):
            assert can_implement(dense, 64, 256, K, "k", 1, fmt), (fmt, K)
        for K in (32, 64, 160, 4032):
            assert can_implement(dense, 64, 256, K, "k", 1, fmt) == (
                fmt == "mxfp4w8a_k64"
            ), (fmt, K)
        assert not can_implement(dense, 64, 256, 4096, "n", 1, fmt)
        ins = dense.instruction(fmt)
        assert "kind::mxf8f6f4.block_scale" in ins and "a=e4m3 b=e2m1" in ins
        swapped = Knobs((128, 16), (1, 1), False, True, "m", 1, 0, False, 1, True)
        assert "b=e2m1 a=e4m3" in swapped.instruction(fmt)
        assert ("(sm_107a)" in ins) == (fmt == "mxfp4w8a_k64")
    assert len(enumerate_space(*REFERENCE_SHAPE, b_major="k", in_fmt="mxfp4w8a")) == 288
    assert (
        len(enumerate_space(*REFERENCE_SHAPE, b_major="k", in_fmt="mxfp4w8a_k64"))
        == 384
    )


def test_traffic_model_counts_each_operand_at_its_width():
    """The dispatcher's traffic terms take the operands' own widths: a weight-streaming shape moves the e2m1 weight at half
    the bytes of the MXFP8 class's and the e4m3 tokens at twice the MXFP4 class's, so the mixed class's bytes sit between the
    two one-format classes'; the swapped record exchanges the widths."""
    from flashinfer.gemm.batch_invariant.heuristic import estimate_cycles

    args = (1, 4096, 4096, 0.5, 2, 128, 128, 1, 32, 1, 148, 1, "m", float("inf"))
    mixed = hbm_traffic(*args, ea=1.0)
    mx4 = hbm_traffic(*args)
    mx8 = hbm_traffic(*args[:3], 1.0, *args[4:])
    assert mx4 < mixed < mx8
    assert hbm_traffic(*args, ea=0.5) == mx4  # the default A width is B's
    k = Knobs((128, 128), (1, 1), False, True, "m", 1)
    for fam in ("sm_100", "sm_107"):
        t = {
            f: estimate_cycles(k, 1, 4096, 4096, 1, "k", f, fam, None, nominal=True)
            for f in ("mxfp4", "mxfp4w8a", "mxfp8")
        }
        assert t["mxfp4"] <= t["mxfp4w8a"] <= t["mxfp8"], (fam, t)


@pytest.mark.parametrize("family,sm_count", PARTS)
def test_dispatcher_picks_are_members_of_the_space(family, sm_count):
    """R3 membership: every configuration the dispatcher or the tuner's candidate list returns for the part's mixed class is
    a well-formed record the kernel accepts for the shape; the candidates are distinct and the default is their first."""
    fmt = mx_class_for("mxfp4w8a", family)
    for M in (1, 8, 16, 64, 300, 4096):
        for N, K in [
            (4096, 4096),
            (14336, 4096),
            (128, 4096),
            (64, 8192),
            (2048, 16384),
            (128, 8192),
            (512, 3968),
        ]:
            C = chunks_for(N, K, fmt)
            top = rank_topk(M, N, K, C, "k", sm_count, 10, family, fmt)
            assert 1 <= len(top) <= 10 and len(set(top)) == len(top)
            assert top[0] == default_config(M, N, K, C, "k", sm_count, family, fmt)
            space = set(enumerate_space(M, N, K, chunks=C, b_major="k", in_fmt=fmt))
            for cfg in top:
                assert cfg.well_formed(fmt) and can_implement(
                    cfg, M, N, K, "k", C, fmt
                ), cfg.describe()
                assert cfg in space, cfg.describe()


def test_unsupported_device_raises_before_any_kernel():
    a = torch.zeros(4, 128).to(E4)
    w = torch.zeros(128, 64, dtype=torch.uint8).t()
    sf = torch.zeros(512, dtype=torch.uint8)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="gemm.mxfp4w8a"):
        mm_mxfp8_mxfp4_batch_invariant(a, w, sf, sf)


def test_public_entry_exists_under_the_library_naming_and_routes_to_the_class():
    """``flashinfer.gemm.mm_mxfp8_mxfp4`` (FlashInfer's ``mm_<activation>_<weight>`` naming) is exported from ``flashinfer`` too,
    with ``mm_mxfp8``'s argument names and order plus the keyword-only ``batch_invariant`` / ``enable_pdl``; the former
    internal name is gone from the API module and the class table, whose key is the public API's; both flag values reach the
    class (a CPU operand raises the class's own error before any kernel, under either value)."""
    fn = flashinfer.gemm.mm_mxfp8_mxfp4
    assert flashinfer.mm_mxfp8_mxfp4 is fn
    assert "mm_mxfp8_mxfp4" in flashinfer.gemm.__all__
    params = inspect.signature(fn).parameters
    assert list(params) == [
        "a",
        "b",
        "a_descale",
        "b_descale",
        "alpha",
        "out_dtype",
        "out",
        "batch_invariant",
        "enable_pdl",
    ]
    assert params["alpha"].default is None
    assert params["out_dtype"].default is torch.bfloat16
    assert params["out"].default is None
    assert params["batch_invariant"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["batch_invariant"].default is False
    assert params["enable_pdl"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["enable_pdl"].default is None
    from flashinfer.gemm.batch_invariant import api

    assert not hasattr(api, "mm_mxfp4w8a_mxfp8a_batch_invariant")
    assert "mm_mxfp4w8a_mxfp8a" not in bi.CLASSES
    assert bi.class_name("mm_mxfp8_mxfp4") == "gemm.mxfp4w8a"
    a = torch.zeros(4, 128).to(E4)
    w = torch.zeros(128, 64, dtype=torch.uint8).t()
    sf = torch.zeros(512, dtype=torch.uint8)
    for flag in (False, True):
        with pytest.raises(bi.BatchInvariantUnsupportedError, match="gemm.mxfp4w8a"):
            fn(a, w, sf, sf, batch_invariant=flag)
    # The public mixed GEMM accepts BF16 and FP16 output.
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="bf16 or fp16"):
        fn(a, w, sf, sf, out_dtype=torch.float32)


# ----------------------------------------------------------------------------------------------- on the device


@needs_device
@pytest.mark.parametrize("M,N,K", [(300, 4096, 4096), (37, 128, 8192), (8, 4096, 4096)])
def test_determinism_and_graph_replay(M, N, K):
    a, sfa = adv_mx8(M, K, 21)
    b, sfb = weight(N, K, 22)
    alpha = alpha_t(0.37)
    ref = bits(mm(a, b, sfa, sfb, alpha)).clone()
    for _ in range(3):
        assert torch.equal(bits(mm(a, b, sfa, sfb, alpha)), ref)
    out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        mm(
            a, b, sfa, sfb, alpha, out=out
        )  # warm-up on the side stream: the compile happens outside the capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            mm(a, b, sfa, sfb, alpha, out=out)
    torch.cuda.current_stream().wait_stream(s)
    for _ in range(2):
        out.zero_()
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(bits(out), ref)


@needs_device
@pytest.mark.parametrize("M,N,K", [(8, 4096, 4096), (300, 2048, 4096)])
def test_public_entry_computes_the_class_under_both_flag_values(M, N, K):
    """Both public flag values accept a caller-provided output and use the same arithmetic."""
    a, sfa = adv_mx8(M, K, 25, span=3)
    b, sfb = weight(N, K, 26, span=3)
    alpha = alpha_t(0.37)
    for od in (torch.bfloat16, torch.float16):
        ref = mm(a, b, sfa, sfb, alpha, out_dtype=od)
        assert norm_rel_err(ref, a, sfa, b.t(), sfb, alpha.item()) <= (
            2**-8 if od == torch.bfloat16 else 2**-11
        )
        ref = bits(ref)
        for flag in (False, True):
            out = flashinfer.mm_mxfp8_mxfp4(
                a, b, sfa, sfb, alpha, out_dtype=od, batch_invariant=flag
            )
            assert out.dtype == od and out.shape == (M, N)
            assert torch.equal(bits(out), ref)
            buf = torch.empty(M, N, dtype=od, device="cuda")
            got = flashinfer.mm_mxfp8_mxfp4(
                a, b, sfa, sfb, alpha, od, buf, batch_invariant=flag
            )
            assert got is buf and torch.equal(bits(buf), ref)
            assert torch.equal(
                bits(
                    flashinfer.mm_mxfp8_mxfp4(
                        a,
                        b,
                        sfa,
                        sfb,
                        alpha,
                        out_dtype=od,
                        batch_invariant=flag,
                        enable_pdl=True,
                    )
                ),
                ref,
            )
    # alpha None is 1.0 on the public entry as on the direct one
    assert torch.equal(
        bits(flashinfer.mm_mxfp8_mxfp4(a, b, sfa, sfb)),
        bits(mm(a, b, sfa, sfb, alpha_t(1.0))),
    )


@needs_device
@pytest.mark.parametrize(
    "M,N,K", [(1, 4096, 4096), (16, 2048, 16384), (300, 1024, 4096), (16, 128, 8192)]
)
def test_pdl_variant_is_a_placement(M, N, K):
    """Programmatic dependent launch (``enable_pdl``; the default follows the device): the kernel waits for the preceding
    grid before its first operand load and releases its dependents once its loads are issued -- the same bits as the plain
    launch for the dispatcher's pick, in a stream of back-to-back calls and
    under graph replay of such a chain."""
    a, sfa = adv_mx8(M, K, 31)
    b, sfb = weight(N, K, 32)
    alpha = alpha_t(0.37)
    ref = bits(mm(a, b, sfa, sfb, alpha, enable_pdl=False)).clone()
    outs = [
        mm(a, b, sfa, sfb, alpha, enable_pdl=True) for _ in range(4)
    ]  # back to back: the dependent launches chain
    torch.cuda.synchronize()
    assert all(torch.equal(bits(o), ref) for o in outs)
    out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        mm(
            a, b, sfa, sfb, alpha, out=out, enable_pdl=True
        )  # warm-up outside the capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(3):
                mm(a, b, sfa, sfb, alpha, out=out, enable_pdl=True)
    torch.cuda.current_stream().wait_stream(s)
    for _ in range(2):
        out.zero_()
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(bits(out), ref)


def test_fingerprint_refuses_what_the_operator_refuses():
    """The receipt exists for the weights the operator accepts: K a multiple of the class's granule (128 on the SM100 class,
    32 on SM107), N a multiple of 8."""
    with pytest.raises(ValueError, match="multiple of 128"):
        bi.fingerprint(
            "mm_mxfp8_mxfp4",
            N=4096,
            K=4096 + 64,
            out_dtype=torch.bfloat16,
            arch="sm_100",
        )
    with pytest.raises(ValueError, match="multiple of 32"):
        bi.fingerprint(
            "mm_mxfp8_mxfp4",
            N=4096,
            K=4096 + 16,
            out_dtype=torch.bfloat16,
            arch="sm_107",
        )
    with pytest.raises(ValueError, match="multiple of 8"):
        bi.fingerprint(
            "mm_mxfp8_mxfp4", N=4092, K=4096, out_dtype=torch.bfloat16, arch="sm_100"
        )
    for N, K in ((0, 128), (128, 0)):
        with pytest.raises(ValueError, match="must be positive"):
            bi.fingerprint(
                "mm_mxfp8_mxfp4", N=N, K=K, out_dtype=torch.bfloat16, arch="sm_100"
            )
    assert bi.fingerprint(
        "mm_mxfp8_mxfp4", N=4096, K=4096 + 64, out_dtype=torch.bfloat16, arch="sm_107"
    )


def test_unsupported_errors_advise_no_fallback():
    """The operator runs the class under both values of ``batch_invariant``, so its errors state the contract and never
    advise calling without the flag (the mode's other classes do)."""
    a = torch.zeros(8, 256, dtype=torch.float8_e4m3fn)
    b = torch.zeros(128, 64, dtype=torch.uint8)
    with pytest.raises(bi.BatchInvariantUnsupportedError) as e:
        flashinfer.mm_mxfp8_mxfp4(
            a, b, torch.zeros(1, dtype=torch.uint8), torch.zeros(1, dtype=torch.uint8)
        )
    msg = str(e.value)
    assert "mm_mxfp8_mxfp4" in msg and "both values of batch_invariant" in msg
    assert (
        "regular backends" not in msg and "call without batch_invariant=True" not in msg
    )


@needs_device
def test_invalid_weight_dtype_reports_the_mixed_class():
    """A weight of another dtype on a supported device is refused under the mixed class's name with the operator's own
    hint (the packing check is shared with the fp4 class, whose name must not appear)."""
    a = torch.zeros(8, 256, dtype=torch.float8_e4m3fn, device="cuda")
    b = torch.zeros(128, 64, dtype=torch.bfloat16, device="cuda")
    sf = torch.zeros(1, dtype=torch.uint8, device="cuda")
    with pytest.raises(bi.BatchInvariantUnsupportedError) as e:
        flashinfer.mm_mxfp8_mxfp4(a, b, sf, sf)
    msg = str(e.value)
    assert "gemm.mxfp4w8a" in msg and "packed fp4" in msg
    assert "regular backends" not in msg and "`gemm.fp4`" not in msg


@needs_device
def test_default_alpha_is_visible_to_a_call_on_another_stream():
    """An independent stream sees the same implicit unit scale as the explicit unit scale."""
    from flashinfer.gemm.batch_invariant import cudnn_gemm

    M, N, K = 64, 2048, 4096
    a, sfa = adv_mx8(M, K, 41)
    b, sfb = weight(N, K, 42)
    ref = bits(mm(a, b, sfa, sfb, alpha_t(1.0))).clone()
    busy = torch.randn(4096, 4096, device="cuda")
    torch.cuda.synchronize()

    # The explicit-alpha graph also owns a device unit scalar. Start the first
    # implicit-alpha call with both caches cold.
    cudnn_gemm.build_quantized_graph.cache_clear()
    cudnn_gemm._unit_alpha.cache_clear()
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(s1):
        # Delay any initialization enqueued on the first stream.
        for _ in range(8):
            busy = torch.mm(busy, busy)
        first = mm(a, b, sfa, sfb, None)
    with torch.cuda.stream(s2):
        second = mm(a, b, sfa, sfb, None)
    torch.cuda.synchronize()
    assert torch.equal(bits(first), ref)
    assert torch.equal(bits(second), ref)


@needs_device
def test_pdl_reads_alpha_after_the_dependency_wait():
    """The dependent GEMM reads an FP32 alpha written by the preceding PDL grid."""
    M, N, K = 64, 2048, 4096
    a = torch.zeros(M, K, dtype=torch.uint8, device="cuda")
    a[:, 0] = 0x38  # E4M3 1.0
    w = torch.zeros(N, K // 2, dtype=torch.uint8, device="cuda")
    w[1, 0] = 0x02  # E2M1 1.0 in the low nibble
    sfa = swizzle_sf(torch.full((M, K // 32), 127, dtype=torch.uint8))
    sfb = swizzle_sf(torch.full((N, K // 32), 127, dtype=torch.uint8))
    a2, sfa2 = adv_mx8(M, K, 53)
    b2, sfb2 = weight(N, K, 54)
    ref = bits(mm(a2, b2, sfa2, sfb2, alpha_t(1), enable_pdl=False))
    out1 = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    # The first two BF16 outputs are +0 and 1, whose little-endian bytes encode FP32 1.0.
    alpha = out1.view(torch.float32)[0, :1]
    for _ in range(4):
        out1.zero_()
        mm(a.view(E4), w.t(), sfa, sfb, out=out1, enable_pdl=True)
        out2 = mm(a2, b2, sfa2, sfb2, alpha, enable_pdl=True)
        torch.cuda.synchronize()
        assert torch.equal(bits(out2), ref)


@needs_device
def test_batch_invariance_along_rows():
    """R2: a row's bits do not depend on the number of rows (the dispatcher's M-dependent picks included: the swapped decode
    records at small M, the dense tiles above), its position or the other rows; the parts' picks agree."""
    N, K = 4096, 4096
    a, sfa = adv_mx8(2049, K, 31)
    b, sfb = weight(N, K, 32)
    alpha = alpha_t(0.37)
    full = bits(mm(a, b, sfa, sfb, alpha))
    rows = sf_bytes_of(sfa, 2049, K)
    for M in (1, 2, 7, 16, 32, 37, 63, 64, 65, 128, 256, 512, 1000, 2049):
        sub = a[:M].contiguous()
        assert torch.equal(
            bits(mm(sub, b, swizzle_sf(rows[:M]), sfb, alpha)), full[:M]
        ), f"M={M}"
    # a row in another batch: the row's scales travel with it
    x5, s5 = adv_mx8(5, K, 33)
    x7, s7 = adv_mx8(7, K, 34)
    x = torch.cat([x5, a[100:101], x7])
    sfx = swizzle_sf(
        torch.cat([sf_bytes_of(s5, 5, K), rows[100:101], sf_bytes_of(s7, 7, K)])
    )
    assert torch.equal(bits(mm(x, b, sfx, sfb, alpha))[5], full[100])


@needs_device
@pytest.mark.parametrize("N,K", [(2048, 4096), (128, 8192)])
def test_permutation_and_subbatch_invariance(N, K):
    """R2 in both forms along the rows, for the unsplit (N = 2048) and the split (N = 128, C = 4) class."""
    M = 512
    a, sfa = adv_mx8(M, K, 41)
    b, sfb = weight(N, K, 42)
    alpha = alpha_t(0.37)
    full = bits(mm(a, b, sfa, sfb, alpha))
    rows = sf_bytes_of(sfa, M, K)
    perm = torch.randperm(M, generator=torch.Generator().manual_seed(1))
    assert torch.equal(
        bits(mm(a[perm.cuda()].contiguous(), b, swizzle_sf(rows[perm]), sfb, alpha)),
        full[perm.cuda()],
    )
    for k in (1, 2, 4, 8, 16, 37):
        idx = torch.randperm(M, generator=torch.Generator().manual_seed(k))[:k]
        got = bits(mm(a[idx.cuda()].contiguous(), b, swizzle_sf(rows[idx]), sfb, alpha))
        assert torch.equal(got, full[idx.cuda()]), f"k={k}"


@needs_device
def test_exact_single_product_step():
    """A single nonzero product in each dot checks every finite E4M3 and E2M1 encoding."""
    codes = torch.tensor(
        [c for c in range(256) if c not in (0x7F, 0xFF)], dtype=torch.uint8
    )
    va = codes.view(E4).double()  # the 254 finite e4m3 values
    M, N, K = len(codes), 16, 128
    a = torch.zeros(M, K, dtype=torch.uint8)
    a[:, 0] = codes
    bq = torch.zeros(N, K // 2, dtype=torch.uint8)
    bq[:, 0] = torch.arange(
        16, dtype=torch.uint8
    )  # element 0 of row j is code j (low nibble), element 1 is +0
    for ea, eb in ((127, 127), (97, 147), (167, 117), (77, 67)):
        sfa = swizzle_sf(torch.full((M, K // 32), ea, dtype=torch.uint8))
        sfb = swizzle_sf(torch.full((N, K // 32), eb, dtype=torch.uint8))
        exact = (
            va[:, None] * E2M1_VALUES[None, :] * 2.0 ** (ea + eb - 254)
        ).float() + 0.0
        for dtype in (torch.bfloat16, torch.float16):
            out = mm(a.view(E4).cuda(), bq.cuda().t(), sfa, sfb, out_dtype=dtype)
            assert torch.equal(bits(out).cpu(), bits(exact.to(dtype))), (ea, eb, dtype)


@needs_device
def test_matches_flashinfer_sm107_kernel_on_its_mixed_path():
    """On SM107 FlashInfer's own sm107 block-scaled kernel computes the pair through its pointer entry (``A{e4m3} x B{e2m1}``:
    the same instruction, the packed weight, K ascending, alpha then one conversion): bit-identical to
    ``gemm.mxfp4w8a.kfold64`` at six (shape, tile, cluster) points incl. 2-CTA and 4-CTA clusters, for bf16 / f16 (a
    power-of-two alpha: that kernel rounds alpha to the output type) and f32 (any alpha), and at the B-reuse record."""
    if part_of(torch.device("cuda"))[1] != "sm_107":
        pytest.skip("the SM107 class and FlashInfer's sm107 kernel exist on SM107 only")
    points = [
        (256, 512, 1024, (128, 128), (1, 1)),
        (1000, 4096, 4096, (256, 128), (2, 1)),
        (37, 2048, 3968, (128, 64), (1, 1)),
        (200, 8192, 512, (128, 192), (1, 2)),
        (300, 3072, 2048, (256, 256), (2, 2)),
        (64, 4096, 128, (128, 256), (1, 4)),
    ]
    for i, (M, N, K, tile, cluster) in enumerate(points):
        a, sfa = adv_mx8(M, K, 301 + i, span=6)
        b, sfb = weight(N, K, 331 + i, span=6)
        for od, alpha in (
            (torch.bfloat16, alpha_t(2.0**-4)),
            (torch.float16, alpha_t(2.0**-2)),
        ):
            result = mm(a, b, sfa, sfb, alpha, out_dtype=od)
            oracle = fi_sm107_mixed(a, b.t(), sfa, sfb, alpha, od, tile, cluster)
            assert torch.equal(bits(result), bits(oracle)), (M, N, K, od)


@needs_device
def test_flashinfer_quantizers_feed_the_class():
    """``mxfp8_quantize`` (the activations) and ``mxfp4_quantize`` (the weights, 128x4 swizzled) feed the entry without
    conversion: within the fp64 yardstick of their dequantisation, bit-identical to FlashInfer's sm107 kernel on SM107, and the
    ``float4_e2m1fn_x2`` view of the weight gives the bits of its uint8 view."""
    M, N, K = 300, 1024, 2048
    g = torch.Generator(device="cuda").manual_seed(5)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16, generator=g)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16, generator=g)
    a, sfa = flashinfer.mxfp8_quantize(x, True)
    bq, sfb = flashinfer.mxfp4_quantize(w, sfLayout=flashinfer.SfLayout.layout_128x4)
    sfa, sfb = sfa.view(torch.uint8).reshape(-1), sfb.view(torch.uint8).reshape(-1)
    bu8 = bq.view(torch.uint8)
    alpha = alpha_t(2.0**-3)
    bf = mm(a, bu8.t(), sfa, sfb, alpha)
    assert norm_rel_err(bf, a, sfa, bu8, sfb, 2.0**-3) <= 2.0**-8
    assert torch.equal(bits(mm(a, bq.t(), sfa, sfb, alpha)), bits(bf))
    if part_of(torch.device("cuda"))[1] == "sm_107":
        assert torch.equal(
            bits(bf), bits(fi_sm107_mixed(a, bu8, sfa, sfb, alpha, torch.bfloat16))
        )


@needs_device
def test_k_granules_of_the_two_classes():
    """SM100/103 require K in multiples of 128. SM107 pads the final 64-wide step."""
    if part_of(torch.device("cuda"))[1] != "sm_107":
        for K in (64, 160, 4032):
            a, sfa = adv_mx8(16, K, 401)
            b, sfb = weight(256, K, 402)
            with pytest.raises(
                bi.BatchInvariantUnsupportedError, match="multiple of 128"
            ):
                mm(a, b, sfa, sfb)
        return
    for K, Kp in ((4032, 4096), (160, 256), (64, 128)):
        M, N = 37, 256
        a, sfa = adv_mx8(M, K, 405)
        bt, sfb = adv_mx4(N, K, 406)
        alpha = alpha_t(0.37)
        got = bits(mm(a, bt.t(), sfa, sfb, alpha))
        ap = torch.zeros(M, Kp, dtype=torch.uint8, device="cuda")
        ap[:, :K] = a.view(torch.uint8)
        bp = torch.zeros(N, Kp // 2, dtype=torch.uint8, device="cuda")
        bp[:, : K // 2] = bt
        sa = torch.zeros(M, Kp // 32, dtype=torch.uint8)
        sa[:, : K // 32] = sf_bytes_of(sfa, M, K)
        sb = torch.zeros(N, Kp // 32, dtype=torch.uint8)
        sb[:, : K // 32] = sf_bytes_of(sfb, N, K)
        pad = bits(mm(ap.view(E4), bp.t(), swizzle_sf(sa), swizzle_sf(sb), alpha))
        assert torch.equal(got, pad), K


@needs_device
def test_caller_output_aliasing_an_operand_is_refused_on_the_public_path():
    """Only the entry's own fresh output skips the aliasing check: a caller's output over the activation's bytes is refused
    through ``flashinfer.mm_mxfp8_mxfp4`` as through the direct entry."""
    M, N, K = 16, 4096, 4096
    a, sfa = adv_mx8(M, K, 181)
    b, sfb = weight(N, K, 182)
    ref = bits(mm(a, b, sfa, sfb))
    buf = torch.empty(M * max(2 * N, K), dtype=torch.uint8, device="cuda")
    alias_a = buf[: M * K].view(E4).view(M, K)
    alias_out = buf[: 2 * M * N].view(torch.bfloat16).view(M, N)
    for call in (flashinfer.mm_mxfp8_mxfp4, mm):
        with pytest.raises(
            bi.BatchInvariantUnsupportedError, match="A overlapping the output"
        ):
            call(alias_a, b, sfa, sfb, None, torch.bfloat16, out=alias_out)
    out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    assert flashinfer.mm_mxfp8_mxfp4(a, b, sfa, sfb, None, torch.bfloat16, out) is out
    assert torch.equal(bits(out), ref)


@needs_device
def test_unsupported_inputs_raise_and_nothing_falls_back():
    a, sfa = adv_mx8(16, 4096, 121)
    b, sfb = weight(4096, 4096, 122)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="float8_e4m3fn A"):
        mm(a.view(torch.uint8), b, sfa, sfb)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="packed fp4"):
        mm(a, b.view(torch.int8), sfa, sfb)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="K-major"):
        mm(a, b.contiguous(), sfa, sfb)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="matching K"):
        mm(a, b[:1024], sfa, sfb)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="uint8"):
        mm(a, b, sfa.view(torch.int8), sfb)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="at least"):
        mm(a, b, sfa, sfb[:-512])
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="on A's device"):
        mm(a, b, sfa, sfb.cpu())
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="multiple of 32"):
        mm(a[:, :4080], b[:2040], sfa, sfb)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="2-D operands"):
        mm(a.unsqueeze(0), b, sfa, sfb)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="float32 tensor alpha"):
        mm(a, b, sfa, sfb, 0.5)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="one scalar alpha"):
        mm(a, b, sfa, sfb, torch.ones(16, device="cuda"))
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="bf16 or fp16"):
        mm(a, b, sfa, sfb, out_dtype=torch.int32)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="out"):
        mm(
            a,
            b,
            sfa,
            sfb,
            out=torch.empty(16, 4096, dtype=torch.float16, device="cuda"),
        )
    assert torch.equal(bits(mm(a, b, sfa, sfb)), bits(mm(a, b, sfa, sfb, alpha_t(1.0))))
    out = mm(a[:0], b, swizzle_sf(torch.zeros(0, 128, dtype=torch.uint8)), sfb)
    assert out.shape == (0, 4096)
