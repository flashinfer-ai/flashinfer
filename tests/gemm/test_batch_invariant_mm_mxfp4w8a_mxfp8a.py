# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The batch-invariant mixed MX GEMM: ``flashinfer.gemm.batch_invariant.api.mm_mxfp4w8a_mxfp8a_batch_invariant`` (e2m1
weights against e4m3 activations, UE8M0 block scales per 32 elements on both; FlashInfer has no dense operator for the pair,
so the entry is the mode's own hook).

The classes ``gemm.mxfp4w8a.kfold`` / ``kchunk[C]`` (the SM100 / SM103 ``kind::mxf8f6f4`` instruction with mixed operand
formats: 32 elements per step, one UE8M0 scale per row; the e2m1 weight reaches the instruction through the TMA's unpacking
format, so K is a whole number of 128-element runs) and ``gemm.mxfp4w8a.kfold64`` / ``kchunk64[C]`` (the SM107 instruction:
64 per step, the packed weight) are the dense K fold with a mixed-operand block-scaled step and the caller's ``alpha`` in the
chain. There is no exact CPU oracle for the step's fold, so the classes are checked by the exhaustive single-product sweep
(one non-zero product per element: the e4m3 x e2m1 product is exact in binary32), by bit agreement across their
configurations -- the dense records, the swapped decode records (the operand formats exchanged), the in-cluster split, the
in-CTA chunk fold, the workspace split and (SM107) the B-reuse record -- by the launch-model properties R1-R2 on adversarial
operands and scales, on SM107 by bit equality with FlashInfer's own sm107 block-scaled kernel on its mixed path (the same
instruction, the same K order), and by an accuracy yardstick against the fp64 product of the dequantised operands. The e4m3
x e4m3 class on the e2m1 weight re-coded to e4m3 (an exact re-coding) is a DIFFERENT fold of the same products: pinned here
as an observation, so FlashInfer's MXFP8 backends are not oracles of these classes. Device-free tests cover the fingerprints
(the architecture is a dimension), the chunk rule and the granules, the spaces, the dispatcher's traffic model and its picks;
the GPU tests need SM100 / SM103 / SM107 (the SM100 class runs on SM107 too, the SM107 class nowhere else). Set
``FLASHINFER_BI_FULL_SPACE=1`` for the whole-space configuration agreement.
"""

import dataclasses
import os

import pytest
import torch

import flashinfer
from flashinfer import batch_invariant as bi
from flashinfer.gemm.batch_invariant.api import (
    MM_MXW48_ARCHS,
    mm_mxfp4w8a_mxfp8a_batch_invariant,
    mm_mxfp8_batch_invariant,
    mxfp4w8a_format_for,
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
FULL_SPACE = bool(os.environ.get("FLASHINFER_BI_FULL_SPACE"))
_supported = torch.cuda.is_available() and bi.is_supported_device(
    torch.device("cuda"), MM_MXW48_ARCHS
)
needs_device = pytest.mark.skipif(
    not _supported,
    reason="needs a CUDA device of SM100 / SM103 / SM107 with the CuTe DSL",
)
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


def recode_e2m1_as_e4m3(packed: torch.Tensor) -> torch.Tensor:
    """The packed e2m1 weight ``(rows, K / 2)`` as an e4m3 tensor ``(rows, K)`` of the SAME values (every e2m1 value is an
    e4m3 value: an exact re-coding, a different input to the MXFP8 class)."""
    return unpack_e2m1(packed).to(E4).cuda()


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
    return mm_mxfp4w8a_mxfp8a_batch_invariant(
        a, b, sfa, sfb, alpha, out_dtype=out_dtype, **kw
    )


def alpha_t(v: float) -> torch.Tensor:
    return torch.full((1,), float(v), device="cuda", dtype=torch.float32)


def device_fmt() -> str:
    return mxfp4w8a_format_for(torch.device("cuda"))


def device_fmts():
    """The classes the device runs: both on SM107 (the SM100 instruction exists there too), the SM100 class elsewhere."""
    fam = part_of(torch.device("cuda"))[1]
    return ("mxfp4w8a", "mxfp4w8a_k64") if fam == "sm_107" else ("mxfp4w8a",)


def shapes_of(fmt: str, shapes):
    """The shapes a class takes (the SM100 class: K in whole 128-element runs)."""
    return [s for s in shapes if s[2] % k_granule(fmt) == 0]


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
        "mm_mxfp4w8a_mxfp8a", N=4096, K=4096, out_dtype=torch.bfloat16, arch="sm_107"
    )
    f100 = bi.fingerprint(
        "mm_mxfp4w8a_mxfp8a", N=4096, K=4096, out_dtype=torch.bfloat16, arch="sm_100"
    )
    assert f107.startswith("gemm-mxfp4w8a-k64-kfold-scaled-") and f100.startswith(
        "gemm-mxfp4w8a-kfold-scaled-"
    )
    assert f107 != f100  # two instructions, two classes
    assert f100 == bi.fingerprint(
        "mm_mxfp4w8a_mxfp8a", N=4096, K=4096, out_dtype="bf16", arch="sm_103"
    )
    assert f107 == bi.fingerprint(
        "mm_mxfp4w8a_mxfp8a", N=4096, K=4096, out_dtype="bf16", arch=(10, 7)
    )
    assert bi.fingerprint(
        "mm_mxfp4w8a_mxfp8a", N=128, K=8192, out_dtype=torch.float16, arch="sm_100"
    ).startswith("gemm-mxfp4w8a-kchunk4-scaled-")
    # the output dtype and the raw K are dimensions of the specification; M and the layout of the scale buffers are not
    distinct = {
        bi.fingerprint("mm_mxfp4w8a_mxfp8a", N=4096, K=K, out_dtype=od, arch=arch)
        for K in (4096, 4096 + 128)
        for od in (torch.bfloat16, torch.float16)
        for arch in ("sm_100", "sm_107")
    }
    assert len(distinct) == 8
    # a different class from the one-format MX classes of the same shape and part: the step's operand formats differ
    for arch in ("sm_100", "sm_107"):
        mixed = bi.fingerprint(
            "mm_mxfp4w8a_mxfp8a", N=4096, K=4096, out_dtype=torch.bfloat16, arch=arch
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
    assert bi.class_name("mm_mxfp4w8a_mxfp8a") == "gemm.mxfp4w8a"
    with pytest.raises(TypeError):
        bi.fingerprint("mm_mxfp4w8a_mxfp8a", N=4096, K=4096)
    with pytest.raises(ValueError):
        bi.fingerprint(
            "mm_mxfp4w8a_mxfp8a",
            N=4096,
            K=4096,
            out_dtype=torch.float32,
            arch="sm_100",
        )
    with pytest.raises(ValueError):
        bi.fingerprint(
            "mm_mxfp4w8a_mxfp8a", N=4096, K=4096, out_dtype=torch.bfloat16, arch="sm_90"
        )


def test_chunk_rule_and_granules():
    """One chunk rule for every MX class (128-element granules: a whole scale-factor atom column and whole steps of either
    instruction); the body's K granule is 128 for the SM100 class (the TMA's unpacking runs over the e2m1 row) and 32 for the
    SM107 one (one block scale, 16-byte rows of the packed weight)."""
    for fmt in MX_MIXED_FMTS:
        assert chunk_unit(fmt) == 128
        for N, K in [(4096, 4096), (128, 4096), (64, 8192), (128, 8192), (64, 16640)]:
            assert chunks_for(N, K, fmt) == chunks_for(N, K, "mxfp8")
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
        mm_mxfp4w8a_mxfp8a_batch_invariant(a, w, sf, sf)


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
@pytest.mark.parametrize(
    "M,N,K", [(1, 4096, 4096), (16, 2048, 16384), (300, 1024, 4096), (16, 128, 8192)]
)
def test_pdl_variant_is_a_placement(M, N, K):
    """Programmatic dependent launch (``enable_pdl``; the default follows the device): the kernel waits for the preceding
    grid before its first operand load and releases its dependents once its loads are issued -- the same bits as the plain
    launch (the bf16 output and the fp32 chain input) for the dispatcher's pick, in a stream of back-to-back calls and
    under graph replay of such a chain."""
    a, sfa = adv_mx8(M, K, 31)
    b, sfb = weight(N, K, 32)
    alpha = alpha_t(0.37)
    ref = bits(mm(a, b, sfa, sfb, alpha, enable_pdl=False)).clone()
    acc = bits(
        mm(a, b, sfa, sfb, alpha, out_dtype=torch.float32, enable_pdl=False)
    ).clone()
    outs = [
        mm(a, b, sfa, sfb, alpha, enable_pdl=True) for _ in range(4)
    ]  # back to back: the dependent launches chain
    torch.cuda.synchronize()
    assert all(torch.equal(bits(o), ref) for o in outs)
    assert torch.equal(
        bits(mm(a, b, sfa, sfb, alpha, out_dtype=torch.float32, enable_pdl=True)), acc
    )
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
    # the dispatcher's picks for other row counts and parts compute the same bits
    fmt = device_fmt()
    for M in (16, 1000):
        for cfg in (
            default_config(M, N, K, 1, "k", 204, "sm_107", fmt),
            default_config(2049, N, K, 1, "k", 148, "sm_100", fmt),
            default_config(1, N, K, 1, "k", 148, "sm_100", fmt),
        ):
            if can_implement(cfg, M, N, K, "k", 1, fmt):
                got = bits(
                    mm(a[:M].clone(), b, swizzle_sf(rows[:M]), sfb, alpha, config=cfg)
                )
                assert torch.equal(got, full[:M]), (M, cfg.describe())


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
@pytest.mark.parametrize("M,N,K", [(1000, 4096, 4096), (37, 2048, 3968)])
def test_config_agreement(M, N, K):
    """R3: every configuration the tuner may choose (the dispatcher's top candidates on every part that runs the class) and
    a spread of hand-picked records (1- and 2-CTA MMA, the 64- / 128- / 192- / 256-wide tiles, clusters, both rasters,
    swizzles, the B-reuse tile on SM107) give identical bits."""
    a, sfa = adv_mx8(M, K, 61)
    b, sfb = weight(N, K, 62)
    alpha = alpha_t(0.37)
    for fmt in device_fmts():
        _config_agreement(fmt, a, b, sfa, sfb, alpha, M, N, K)


def _config_agreement(fmt, a, b, sfa, sfb, alpha, M, N, K):
    cfgs = []
    for family, sm in PARTS:
        if mx_class_for("mxfp4w8a", family) == fmt:
            cfgs += rank_topk(M, N, K, 1, "k", sm, 10, family, fmt)
    cfgs += [
        Knobs((128, 64), (1, 1), False, True, "m", 1),
        Knobs((128, 128), (1, 2), False, True, "n", 2),
        Knobs((128, 192), (4, 1), False, True, "m", 8),
        Knobs((128, 256), (2, 1), False, True, "m", 1),
        Knobs((256, 64), (2, 2), True, True, "n", 4),
        Knobs((256, 128), (2, 1), True, True, "m", 1),
        Knobs((256, 192), (2, 2), True, True, "m", 2),
        Knobs((256, 256), (4, 1), True, True, "n", 1),
        Knobs((512, 128), (2, 1), True, True, "m", 1),  # B-reuse (SM107)
        Knobs((512, 256), (2, 2), True, True, "n", 2),
    ]
    cfgs = list(dict.fromkeys(cfgs))
    ref = None
    checked = 0
    for c in cfgs:
        if not can_implement(c, M, N, K, "k", 1, fmt):
            continue
        got = bits(mm(a, b, sfa, sfb, alpha, config=c, in_fmt=fmt))
        if ref is None:
            ref = got
        assert torch.equal(got, ref), f"{fmt}: {c.describe()} differs"
        checked += 1
    assert checked >= 14, fmt


@needs_device
@pytest.mark.skipif(
    not FULL_SPACE,
    reason="set FLASHINFER_BI_FULL_SPACE=1: hundreds of compiles per shape",
)
@pytest.mark.parametrize(
    "M,N,K",
    [
        (64, 4096, 128),
        (64, 4096, 256),
        (200, 8192, 512),
        (37, 2048, 3968),
        (16, 8192, 8192),
        (8, 4096, 4096),
    ],
)
def test_config_agreement_whole_space(M, N, K):
    """The hardware hypothesis over the whole space: every record the shape admits (the dense records, the swapped token
    tiles at the decode shapes, the B-reuse tile on SM107) gives one value; K = one 128-element run isolates the instruction
    variants (one step of the SM100 class is a quarter of it: the lowering's radix)."""
    a, sfa = adv_mx8(M, K, 71)
    b, sfb = weight(N, K, 72)
    alpha = alpha_t(0.37)
    for fmt in device_fmts():
        space = enumerate_space(M, N, K, b_major="k", in_fmt=fmt)
        assert len(space) >= 288, fmt
        ref = bits(mm(a, b, sfa, sfb, alpha, config=space[0], in_fmt=fmt))
        bad = [
            c.describe()
            for c in space[1:]
            if not torch.equal(
                bits(mm(a, b, sfa, sfb, alpha, config=c, in_fmt=fmt)), ref
            )
        ]
        assert not bad, (fmt, bad[:8])
        assert torch.equal(bits(mm(a, b, sfa, sfb, alpha, in_fmt=fmt)), ref)


@needs_device
def test_swapped_decode_records_compute_the_class():
    """The decode specialisation (``swap_ab``: the weights as the kernel's A, the tokens as its B, one token tile) on the
    narrow 8 / 16 / 32 tiles and the 64-wide one, 1- and 2-CTA, clusters along M, every raster: the same bits as the dense
    records (the instruction variant with the operand formats exchanged joins the hardware hypothesis), at M = 1 .. 64 and
    for a bf16 and an fp16 output."""
    fmt = device_fmt()
    N, K = 4096, 4096
    for M, od in (
        (1, torch.bfloat16),
        (5, torch.float16),
        (8, torch.bfloat16),
        (16, torch.float16),
        (33, torch.bfloat16),
        (64, torch.bfloat16),
    ):
        a, sfa = adv_mx8(M, K, 70 + M)
        b, sfb = weight(N, K, 71, span=6 if od == torch.float16 else 12)
        alpha = alpha_t(2.0**-4)
        dense = Knobs((128, 128), (1, 1), False, True, "m", 1)
        ref = bits(mm(a, b, sfa, sfb, alpha, out_dtype=od, config=dense))
        recs = [
            k for k in enumerate_space(M, N, K, b_major="k", in_fmt=fmt) if k.swap_ab
        ]
        pick = {(k.tile_n, k.tile_m, k.cluster_m, k.raster_along): k for k in recs}
        assert len(pick) >= 4, M
        for k in list(pick.values())[::3] + [k for k in recs if k.tile_n == 64][:4]:
            got = bits(mm(a, b, sfa, sfb, alpha, out_dtype=od, config=k))
            assert torch.equal(got, ref), (M, k.describe())
        assert torch.equal(
            bits(mm(a, b, sfa, sfb, alpha, out_dtype=od)), ref
        )  # the dispatcher's own pick


@needs_device
@pytest.mark.parametrize(
    "M,N,K,od",
    [
        (37, 128, 8192, torch.bfloat16),
        (300, 128, 16384, torch.float16),
        (16, 128, 4096, torch.bfloat16),
    ],
)
def test_split_implementations_agree(M, N, K, od):
    """The split class ``kchunk[C]`` has three implementations: the workspace split (chunks on different CTAs, an fp32
    workspace, a fixup fold in chunk order and the chain), the in-CTA chunk fold (one CTA folds every chunk of its tile
    through a TMEM running sum) and the in-cluster split (the chunks on the CTAs of one cluster, the fold through
    distributed shared memory). They are one term in Lean (``ctaSplitMXW48_eq_kchunkMXW48``,
    ``clusterSplitMXW48_eq_kchunkMXW48`` and the K64 twins) and one bit pattern here; and the split class is a different
    specification from the unsplit fold (expected)."""
    fmt = device_fmt()
    C = chunks_for(N, K, fmt)
    assert C > 1
    span = (
        12 if od == torch.bfloat16 else 2
    )  # an fp16 output stays finite over the long K
    a, sfa = adv_mx8(M, K, 81, span=span)
    b, sfb = weight(N, K, 82, span=span)
    alpha = alpha_t(0.37)
    ws = Knobs((128, 64), (1, 1), False, True, "m", 1)
    ref = bits(mm(a, b, sfa, sfb, alpha, out_dtype=od, config=ws, chunks=C))
    space = enumerate_space(M, N, K, chunks=C, b_major="k", in_fmt=fmt)
    recs = [
        k
        for k in rank_topk(M, N, K, C, "k", 204, 10, "sm_107", fmt)
        if k.cta_chunks or k.cluster_k > 1
    ]
    recs += [k for k in space if k.cluster_k > 1][::7]
    recs += [k for k in space if k.cta_chunks][::29]
    recs += [
        Knobs((128, 64), (1, 4), False, True, "m", 1, 0, True),
        Knobs((256, 128), (2, 1), True, True, "n", 2, 0, True),
        Knobs((128, 128), (1, 1), False, True, "m", 1, 0, False, 2),
        Knobs((128, 64), (1, 1), False, True, "n", 1, 0, False, 4),
    ]
    recs = [k for k in dict.fromkeys(recs) if can_implement(k, M, N, K, "k", C, fmt)]
    assert any(k.cta_chunks for k in recs)
    assert C not in (2, 4) or any(
        k.cluster_k > 1 for k in recs
    )  # the in-cluster split exists at C = 2 / 4
    for k in recs:
        got = bits(mm(a, b, sfa, sfb, alpha, out_dtype=od, config=k, chunks=C))
        assert torch.equal(got, ref), k.describe()
    assert torch.equal(
        bits(mm(a, b, sfa, sfb, alpha, out_dtype=od)), ref
    )  # the dispatcher's own pick is a member too
    # a different specification from the unsplit fold (on the fp32 accumulator: a 16-bit output may round the two folds alike)
    acc = bits(mm(a, b, sfa, sfb, alpha, out_dtype=torch.float32, config=ws, chunks=C))
    assert not torch.equal(
        bits(mm(a, b, sfa, sfb, alpha, out_dtype=torch.float32, config=ws, chunks=1)),
        acc,
    )


@needs_device
@pytest.mark.parametrize(
    "M,N,K,C",
    [
        (1, 4096, 4096, 4),
        (16, 2048, 16384, 4),
        (8, 128, 8192, 4),
        (37, 4096, 32768, 8),
        (64, 512, 32768, 4),
    ],
)
def test_swapped_split_records_agree(M, N, K, C):
    """The swapped records of the split classes (the exchanged operands' chunk partials through the workspace, then the same
    fixup): every swapped record of the split shape's space (sampled) gives the dense workspace split's bits (bf16 output and
    fp32 accumulator); the token tile holds all M rows; sub-batches reproduce their rows; the dispatcher's pick is a member."""
    for fmt in device_fmts():
        if K % k_granule(fmt) or (K // C) % k_granule(fmt):
            continue
        a, sfa = adv_mx8(M, K, 91)
        b, sfb = weight(N, K, 92)
        alpha = alpha_t(0.37)
        ws = Knobs((128, 64), (1, 1), False, True, "m", 1)
        ref = bits(mm(a, b, sfa, sfb, alpha, config=ws, chunks=C, in_fmt=fmt))
        acc_ref = bits(
            mm(
                a,
                b,
                sfa,
                sfb,
                alpha,
                out_dtype=torch.float32,
                config=ws,
                chunks=C,
                in_fmt=fmt,
            )
        )
        space = enumerate_space(M, N, K, chunks=C, b_major="k", in_fmt=fmt)
        recs = [k for k in space if k.swap_ab]
        assert recs and all(
            k.cluster_k == 1 and not k.cta_chunks and k.tile_n >= M for k in recs
        ), fmt
        for k in recs[::5] + recs[-1:]:
            assert torch.equal(
                bits(mm(a, b, sfa, sfb, alpha, config=k, chunks=C, in_fmt=fmt)), ref
            ), (fmt, k.describe())
            assert torch.equal(
                bits(
                    mm(
                        a,
                        b,
                        sfa,
                        sfb,
                        alpha,
                        out_dtype=torch.float32,
                        config=k,
                        chunks=C,
                        in_fmt=fmt,
                    )
                ),
                acc_ref,
            ), (fmt, k.describe())
        for fam, sm in (
            ("sm_107", 212),
            ("sm_100", 148),
        ):  # the dispatcher's pick at the split shape is a member of the space
            assert default_config(M, N, K, C, "k", sm, fam, fmt) in set(space), (
                fmt,
                fam,
            )
        if M > 1:
            idx = torch.randperm(M, generator=torch.Generator().manual_seed(7))[
                : max(1, M // 2)
            ]
            rows = sf_bytes_of(sfa, M, K)
            sub = bits(
                mm(
                    a[idx.cuda()].contiguous(),
                    b,
                    swizzle_sf(rows[idx]),
                    sfb,
                    alpha,
                    config=recs[0],
                    chunks=C,
                    in_fmt=fmt,
                )
            )
            assert torch.equal(sub, ref[idx.cuda()])


@needs_device
def test_persistent_grid_size_does_not_change_bits():
    """R2 for the SM count: a smaller persistent grid is what a smaller part, an SM carveout, MPS or a green context
    produce; it only decides which CTA computes which tile (and, for the split class, which chunk)."""
    fmt = device_fmt()
    for M, N, K in [(1000, 4096, 4096), (64, 64, 65536)]:
        a, sfa = adv_mx8(M, K, 91)
        b, sfb = weight(N, K, 92)
        alpha = alpha_t(0.37)
        C = chunks_for(N, K, fmt)
        base = default_config(M, N, K, C, "k", 204, "sm_107", fmt)
        ref = bits(mm(a, b, sfa, sfb, alpha, config=base, chunks=C))
        for g in (1, 3, 17, 74):
            cfg = dataclasses.replace(base, grid_clusters=g)
            assert torch.equal(
                bits(mm(a, b, sfa, sfb, alpha, config=cfg, chunks=C)), ref
            ), f"grid={g} at {(M, N, K)}"


@needs_device
def test_exact_single_product_step():
    """The step's per-element function, measured exhaustively: one non-zero product per element (every finite e4m3 code
    against every e2m1 code, at block scales spanning the UE8M0 range and keeping the products normal) is the exact product
    in binary32 (the e4m3 x e2m1 product has at most six significant bits; a zero product is +0: the accumulator starts at +0
    and +0 + (-0) = +0), identical across the dense records, the 2-CTA pair and the swapped record, on either class the part
    runs."""
    fam = part_of(torch.device("cuda"))[1]
    fmts = ("mxfp4w8a", "mxfp4w8a_k64") if fam == "sm_107" else ("mxfp4w8a",)
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
    swapped = Knobs((128, 64), (1, 1), False, True, "m", 1, 0, False, 1, True)
    for fmt in fmts:
        cfgs = [
            Knobs((128, 128), (1, 1), False, True, "m", 1),
            Knobs((256, 128), (2, 1), True, True, "n", 2),
            Knobs((128, 64), (1, 2), False, True, "m", 1),
        ]
        for ea, eb in (
            (127, 127),
            (127 - 30, 127 + 20),
            (127 + 40, 127 - 10),
            (127 - 50, 127 - 60),
        ):
            sfa = swizzle_sf(torch.full((M, K // 32), ea, dtype=torch.uint8))
            sfb = swizzle_sf(torch.full((N, K // 32), eb, dtype=torch.uint8))
            exact = (
                va.view(-1, 1) * E2M1_VALUES.view(1, -1) * 2.0 ** (ea + eb - 254)
            ).float() + 0.0  # the fold's zero is +0
            outs = [
                mm(
                    a.view(E4).cuda(),
                    bq.cuda().t(),
                    sfa,
                    sfb,
                    alpha_t(1.0),
                    out_dtype=torch.float32,
                    in_fmt=fmt,
                    config=c,
                )
                for c in cfgs
            ]
            assert torch.equal(bits(outs[0].cpu()), bits(exact)), (
                fmt,
                ea,
                eb,
                int((bits(outs[0].cpu()) != bits(exact)).sum()),
            )
            for c, o in zip(cfgs[1:], outs[1:], strict=True):
                assert torch.equal(bits(o), bits(outs[0])), (fmt, ea, eb, c.describe())
            # the swapped record (the operand formats exchanged) holds every token in one tile: the rows in blocks of 64
            for r0 in range(0, M, 64):
                rows = slice(r0, min(r0 + 64, M))
                sfr = swizzle_sf(
                    torch.full((rows.stop - r0, K // 32), ea, dtype=torch.uint8)
                )
                o = mm(
                    a[rows].view(E4).cuda(),
                    bq.cuda().t(),
                    sfr,
                    sfb,
                    alpha_t(1.0),
                    out_dtype=torch.float32,
                    in_fmt=fmt,
                    config=swapped,
                )
                assert torch.equal(bits(o), bits(outs[0][rows])), (fmt, ea, eb, r0)


@needs_device
@pytest.mark.parametrize(
    "M,N,K", [(256, 512, 1024), (1000, 4096, 4096), (16, 128, 8192)]
)
def test_recoded_weight_is_a_different_fold(M, N, K):
    """The e2m1 weight re-coded to e4m3 (an exact re-coding: every e2m1 value is an e4m3 value) through the MXFP8 class of
    the same instruction kind and K step is NOT the mixed class: the same products -- one non-zero product per row agrees
    bit for bit, the step's per-element function is the exact product in both -- and both accumulators within the fp64
    yardstick, but the fp32 accumulators differ (a last-place difference at some outputs; observed on SM100 and SM107 at
    every shape of this test, the split shape included). The instruction folds a 4-bit operand's products differently from
    an 8-bit operand's, so the mixed classes are their own folds (the distinct steps and fingerprints are necessary) and
    FlashInfer's MXFP8 backends (bit-identical to the MXFP8 classes) are not oracles of them. An observation, not a contract:
    the mixed classes promise their own bits only."""
    fmt = device_fmt()
    mx8 = mx_class_for("mxfp8", part_of(torch.device("cuda"))[1])
    a, sfa = adv_mx8(M, K, 201)
    b, sfb = weight(N, K, 202)
    bt = b.t()
    b8 = recode_e2m1_as_e4m3(bt).t()
    one = alpha_t(1.0)
    k = Knobs((128, 128), (1, 1), False, True, "m", 1)
    C = chunks_for(N, K, fmt)
    ours = mm(a, b, sfa, sfb, one, out_dtype=torch.float32, config=k, chunks=C)
    rec = mm_mxfp8_batch_invariant(
        a, b8, sfa, sfb, out_dtype=torch.float32, config=k, chunks=C
    )
    assert (
        norm_rel_err(ours, a, sfa, bt, sfb) <= 2.0**-16
        and norm_rel_err(rec, a, sfa, bt, sfb) <= 2.0**-16
    )
    assert not torch.equal(bits(ours), bits(rec)), (fmt, mx8)
    assert not torch.equal(
        bits(mm(a, b, sfa, sfb, one, out_dtype=torch.float32)),
        bits(mm_mxfp8_batch_invariant(a, b8, sfa, sfb, out_dtype=torch.float32)),
    )
    # one non-zero product per row: exact in both steps
    rows = torch.arange(M, device="cuda")
    a1 = torch.zeros_like(a.view(torch.uint8))
    a1[rows, rows % K] = a.view(torch.uint8)[rows, rows % K]
    a1 = a1.view(E4)
    ours1 = mm(a1, b, sfa, sfb, one, out_dtype=torch.float32, config=k, chunks=C)
    rec1 = mm_mxfp8_batch_invariant(
        a1, b8, sfa, sfb, out_dtype=torch.float32, config=k, chunks=C
    )
    assert (
        torch.equal(bits(ours1), bits(rec1))
        and norm_rel_err(ours1, a1, sfa, bt, sfb) <= 2.0**-20
    )


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
            (torch.float32, alpha_t(0.37)),
        ):
            cfg = Knobs(tile, cluster, tile[0] == 256, True, "m", 1)
            ours = mm(
                a, b, sfa, sfb, alpha, out_dtype=od, config=cfg, in_fmt="mxfp4w8a_k64"
            )
            theirs = fi_sm107_mixed(a, b.t(), sfa, sfb, alpha, od, tile, cluster)
            assert torch.equal(bits(ours), bits(theirs)), (M, N, K, tile, cluster, od)
    # the B-reuse record (two instructions per K step on one B fragment) against the plain record and the kernel
    a, sfa = adv_mx8(1000, 4096, 341, span=6)
    b, sfb = weight(4096, 4096, 342, span=6)
    alpha = alpha_t(2.0**-4)
    plain = bits(
        mm(
            a,
            b,
            sfa,
            sfb,
            alpha,
            config=Knobs((256, 128), (2, 1), True, True, "m", 1),
            in_fmt="mxfp4w8a_k64",
        )
    )
    breuse = bits(
        mm(
            a,
            b,
            sfa,
            sfb,
            alpha,
            config=Knobs((512, 128), (2, 1), True, True, "m", 1),
            in_fmt="mxfp4w8a_k64",
        )
    )
    assert torch.equal(plain, breuse)
    assert torch.equal(
        plain,
        bits(
            fi_sm107_mixed(
                a, b.t(), sfa, sfb, alpha, torch.bfloat16, (256, 128), (2, 1)
            )
        ),
    )


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
    out = mm(a, bu8.t(), sfa, sfb, alpha, out_dtype=torch.float32)
    assert norm_rel_err(out, a, sfa, bu8, sfb, 2.0**-3) <= 2.0**-16
    bf = mm(a, bu8.t(), sfa, sfb, alpha)
    assert torch.equal(bits(bf), bits(out.to(torch.bfloat16)))
    assert torch.equal(bits(mm(a, bq.t(), sfa, sfb, alpha)), bits(bf))
    if part_of(torch.device("cuda"))[1] == "sm_107":
        assert torch.equal(
            bits(bf), bits(fi_sm107_mixed(a, bu8, sfa, sfb, alpha, torch.bfloat16))
        )


@needs_device
def test_k_granules_of_the_two_classes():
    """The SM100 class takes K in whole 128-element runs (the TMA's unpacking format over the e2m1 row): other K is rejected,
    not padded. The SM107 class pads its last step as the other MX classes do: K = 4032 (a half-padded stage: one padded
    64-wide step) equals its explicit zero padding to 4096 (values +0, scale byte 0x00), K = 160 its padding to 256."""
    fam = part_of(torch.device("cuda"))[1]
    for K in (64, 160, 4032):
        a, sfa = adv_mx8(16, K, 401)
        b, sfb = weight(256, K, 402)
        with pytest.raises(bi.BatchInvariantUnsupportedError, match="multiple of 128"):
            mm(a, b, sfa, sfb, in_fmt="mxfp4w8a")
    if fam != "sm_107":
        with pytest.raises(
            bi.BatchInvariantUnsupportedError, match="mxfp4w8a_k64 on SM107"
        ):
            a, sfa = adv_mx8(16, 4096, 403)
            b, sfb = weight(256, 4096, 404)
            mm(a, b, sfa, sfb, in_fmt="mxfp4w8a_k64")
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
def test_the_two_instructions_are_two_classes():
    """On SM107 both instructions exist: the SM100 class (``in_fmt="mxfp4w8a"``) runs there too and is a different
    specification from the device's own class (its fold updates the accumulator every 32 elements, not every 64); each
    agrees with itself across its configurations."""
    fam = part_of(torch.device("cuda"))[1]
    a, sfa = adv_mx8(300, 4096, 93)
    b, sfb = weight(2048, 4096, 94)
    alpha = alpha_t(0.37)
    if fam != "sm_107":
        assert device_fmt() == "mxfp4w8a"
        return
    assert device_fmt() == "mxfp4w8a_k64"
    k64 = bits(mm(a, b, sfa, sfb, alpha, out_dtype=torch.float32))
    k32 = bits(mm(a, b, sfa, sfb, alpha, out_dtype=torch.float32, in_fmt="mxfp4w8a"))
    assert not torch.equal(k64, k32)
    for fmt, ref in (("mxfp4w8a", k32), ("mxfp4w8a_k64", k64)):
        for c in rank_topk(300, 2048, 4096, 1, "k", 204, 4, "sm_107", fmt):
            assert torch.equal(
                bits(
                    mm(
                        a,
                        b,
                        sfa,
                        sfb,
                        alpha,
                        out_dtype=torch.float32,
                        config=c,
                        in_fmt=fmt,
                    )
                ),
                ref,
            ), (
                fmt,
                c.describe(),
            )


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
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="chunks = 0"):
        mm(a, b, sfa, sfb, chunks=0)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="granules"):
        mm(a, b, sfa, sfb, chunks=64)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="enumerated space"):
        mm(
            a,
            b,
            sfa,
            sfb,
            config=Knobs((128, 8), (1, 1), False, True, "m", 1, 0, False, 1, True),
            chunks=1,
        )
    # the entry exposes the fp32 chain input for tests; alpha None is 1.0; nothing to launch at M = 0
    f32 = mm(a, b, sfa, sfb, alpha_t(0.37), out_dtype=torch.float32)
    assert torch.equal(
        bits(f32.to(torch.bfloat16)), bits(mm(a, b, sfa, sfb, alpha_t(0.37)))
    )
    assert torch.equal(bits(mm(a, b, sfa, sfb)), bits(mm(a, b, sfa, sfb, alpha_t(1.0))))
    out = mm(a[:0], b, swizzle_sf(torch.zeros(0, 128, dtype=torch.uint8)), sfb)
    assert out.shape == (0, 4096)
