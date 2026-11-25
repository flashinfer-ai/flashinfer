# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Batch-invariant MXFP4 and NVFP4 GEMM: dequantized references, output scaling and row invariance."""

from unittest.mock import patch

import pytest
import torch

import flashinfer
from flashinfer import batch_invariant as bi
from flashinfer.gemm import gemm_base
from flashinfer.autotuner import autotune
from flashinfer.gemm.batch_invariant.api import (
    fp4_format_for,
    mm_fp4_batch_invariant,
)
from flashinfer.gemm.batch_invariant.blocks.operands import can_implement
from flashinfer.gemm.batch_invariant.generator import (
    REFERENCE_SHAPE,
    enumerate_space,
    knob_product,
)
from flashinfer.gemm.batch_invariant.heuristic import FAMILY_OF_CC, PARTS
from flashinfer.gemm.batch_invariant.schedule import (
    chunks_for,
    default_config,
    fp4_class_for,
    mx_class_for,
    rank_topk,
)
from flashinfer.gemm.batch_invariant.spec import (
    FP4_FMTS,
    MXFP4_FMTS,
    NVFP4_FMTS,
    chunk_unit,
)

E4 = torch.float8_e4m3fn
FP4X2 = getattr(torch, "float4_e2m1fn_x2", None)
needs_device = pytest.mark.usefixtures("cudnn_batch_invariant_device")
both_formats = pytest.mark.parametrize("nv", [True, False], ids=["nvfp4", "mxfp4"])
#: The e2m1 values by nibble code (sign in bit 3).
E2M1 = torch.tensor(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
    dtype=torch.float64,
)


def bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16) if t.element_size() == 2 else t.view(torch.int32)


def block(nv: bool) -> int:
    return 16 if nv else 32


def _sf_offsets(rows: int, kb: int):
    """The byte offset of (row, block) in the 128x4 swizzled layout of a (rows, kb) scale matrix: atoms of 128 rows x 4
    blocks (512 bytes; row r, block j of an atom at byte (r % 32) * 16 + (r // 32) * 4 + j), K-atoms consecutive, row-atoms
    strided by the K-atoms of the whole K."""
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
    return rt, kt, off


def swizzle_sf(sf_rows: torch.Tensor) -> torch.Tensor:
    """The (rows, K / block) uint8 block scales as the flat 128x4 swizzled buffer ``nvfp4_quantize`` / ``mxfp4_quantize``
    write (padding bytes zero), on the device."""
    rows, kb = sf_rows.shape
    rt, kt, off = _sf_offsets(rows, kb)
    buf = torch.zeros(rt * 128, kt * 4, dtype=torch.uint8)
    buf[:rows, :kb] = sf_rows.cpu()
    flat = torch.empty(rt * 128 * kt * 4, dtype=torch.uint8)
    flat[off.reshape(-1)] = buf.reshape(-1)
    return flat.cuda()


def sf_values(sf_rows: torch.Tensor, nv: bool) -> torch.Tensor:
    """The scale values of a (rows, kb) uint8 matrix: e4m3 (NVFP4) or 2^(byte - 127) (MXFP4), in fp64."""
    if nv:
        return sf_rows.view(E4).double()
    return torch.exp2(sf_rows.double() - 127)


def adv_fp4(rows: int, K: int, seed: int, nv: bool, span=None):
    """An adversarial 4-bit operand: every e2m1 code with random signs, zeros of both signs, and block scales spread over
    orders of magnitude (e4m3 exponents 7 +- span with random mantissas for NVFP4, UE8M0 127 +- span for MXFP4; span 6 by
    default: the products of neighbouring blocks differ by orders of magnitude yet every block stays within the fp32
    mantissa's reach of the largest, so an accumulation-order difference reaches the output bits; a wider span lets one
    block dominate a row's sum, a narrow one makes the 4-bit fold exact in fp32). Returns the packed (rows, K / 2) uint8 tensor, its scales in the 128x4 swizzled layout, and the (rows, K / block)
    scale matrix (for re-swizzling a row subset). ``span``: the scales' exponent range."""
    g = torch.Generator().manual_seed(seed)
    nib = torch.randint(0, 16, (rows, K), generator=g)
    z = torch.randint(0, 31, (rows, K), generator=g)
    nib[z == 0] = 0
    nib[z == 1] = 8  # -0
    packed = (nib[:, 0::2] | (nib[:, 1::2] << 4)).to(torch.uint8)
    kb = K // block(nv)
    if nv:
        s = 6 if span is None else span
        e = torch.randint(7 - s, 7 + s + 1, (rows, kb), generator=g)
        m = torch.randint(0, 8, (rows, kb), generator=g)
        sf = ((e << 3) | m).to(torch.uint8)
    else:
        s = 6 if span is None else span
        sf = torch.randint(127 - s, 127 + s + 1, (rows, kb), generator=g).to(
            torch.uint8
        )
    return packed.cuda(), swizzle_sf(sf), sf


def weight(N: int, K: int, seed: int, nv: bool, span=None):
    """``mm_fp4``'s weight: the (K / 2, N) transpose of a row-major (N, K / 2) packed tensor and its scales."""
    w, sf, rows = adv_fp4(N, K, seed, nv, span=span)
    return w.t(), sf, rows


def alpha_of(x: float) -> torch.Tensor:
    return torch.tensor([x], dtype=torch.float32, device="cuda")


def bi_mm(a, b, sfa, sfb, alpha=None, nv=True, out_dtype=torch.bfloat16, **kw):
    return flashinfer.mm_fp4(
        a,
        b,
        sfa,
        sfb,
        alpha,
        out_dtype,
        kw.pop("out", None),
        block_size=block(nv),
        backend="auto",
        use_nvfp4=nv,
        batch_invariant=True,
        **kw,
    )


def direct(a, b, sfa, sfb, alpha=None, nv=True, **kw):
    return mm_fp4_batch_invariant(
        a, b, sfa, sfb, alpha, use_nvfp4=nv, block_size=block(nv), **kw
    )


def dequant(q: torch.Tensor, sf_rows: torch.Tensor, nv: bool) -> torch.Tensor:
    rows, Kh = q.shape
    qc = q.cpu()
    nib = torch.empty(rows, 2 * Kh, dtype=torch.long)
    nib[:, 0::2] = (qc & 0x0F).long()
    nib[:, 1::2] = (qc >> 4).long()
    return E2M1[nib] * sf_values(sf_rows, nv).repeat_interleave(block(nv), dim=1)


def norm_rel_err(c, a, sfa_rows, bt, sfb_rows, alpha, nv) -> float:
    """The max norm-relative error against the exact product of the dequantised operands times alpha."""
    A, B = dequant(a, sfa_rows, nv), dequant(bt, sfb_rows, nv)
    al = float(alpha.item()) if alpha is not None else 1.0
    ref = (A @ B.t()) * al
    scale = (A.abs() @ B.abs().t()) * abs(al)
    return ((c.double().cpu() - ref).abs() / (scale + 1e-30)).max().item()


# ----------------------------------------------------------------------------------------------- device-free


def test_fingerprint_is_a_function_of_the_weight_shape_the_output_format_the_format_and_the_architecture():
    fp = bi.fingerprint
    n107 = fp("mm_fp4", N=4096, K=4096, out_dtype=torch.bfloat16, arch="sm_107")
    n100 = fp("mm_fp4", N=4096, K=4096, out_dtype=torch.bfloat16, arch="sm_100")
    assert n107.startswith("gemm-nvfp4-k128-kfold-scaled-") and n100.startswith(
        "gemm-nvfp4-kfold-scaled-"
    )
    assert n107 != n100  # two instructions, two classes
    assert n100 == fp("mm_fp4", N=4096, K=4096, out_dtype="bf16", arch="sm_103")
    assert n100 == fp(
        "mm_fp4", N=4096, K=4096, out_dtype="bf16", arch="sm_100", use_nvfp4=True
    )
    assert n107 == fp("mm_fp4", N=4096, K=4096, out_dtype="bf16", arch=(10, 7))
    # the SM107 NVFP4 class rule: a weight shape the chunk rule splits runs the 64-wide step's class on SM107 too, so its
    # fingerprint is the SM100 one
    s107 = fp("mm_fp4", N=128, K=16384, out_dtype=torch.bfloat16, arch="sm_107")
    assert s107.startswith("gemm-nvfp4-kchunk8-scaled-")
    assert s107 == fp("mm_fp4", N=128, K=16384, out_dtype=torch.bfloat16, arch="sm_100")
    # MXFP4: its own classes, the architecture a dimension at every shape
    m107 = fp(
        "mm_fp4",
        N=4096,
        K=4096,
        out_dtype=torch.bfloat16,
        use_nvfp4=False,
        arch="sm_107",
    )
    m100 = fp(
        "mm_fp4",
        N=4096,
        K=4096,
        out_dtype=torch.bfloat16,
        use_nvfp4=False,
        arch="sm_100",
    )
    assert m107.startswith("gemm-mxfp4-k128-kfold-scaled-") and m100.startswith(
        "gemm-mxfp4-kfold-scaled-"
    )
    assert fp(
        "mm_fp4",
        N=128,
        K=16384,
        out_dtype=torch.float16,
        use_nvfp4=False,
        arch="sm_107",
    ).startswith("gemm-mxfp4-k128-kchunk8-scaled-")
    # the output dtype, the raw K, the format and the architecture are dimensions; M and the scale buffers' layout are not
    distinct = {
        fp("mm_fp4", N=4096, K=K, out_dtype=od, use_nvfp4=nv, arch=arch)
        for K in (4096, 4096 + 32)
        for od in (torch.bfloat16, torch.float16)
        for nv in (True, False)
        for arch in ("sm_100", "sm_107")
    }
    assert len(distinct) == 16
    assert n100 != fp(
        "mm_mxfp8", N=4096, K=4096, out_dtype=torch.bfloat16, arch="sm_100"
    )
    assert bi.class_name("mm_fp4") == "gemm.fp4"
    if torch.cuda.is_available():
        # the default architecture is the current device's: on a part of the mode its fingerprint; on any other part the
        # typed error, because the family selects the block-scaled instruction and no other part's class is borrowed
        if torch.cuda.get_device_capability() in FAMILY_OF_CC:
            assert fp("mm_fp4", N=4096, K=4096, out_dtype=torch.bfloat16) == fp(
                "mm_fp4",
                N=4096,
                K=4096,
                out_dtype=torch.bfloat16,
                arch=torch.device("cuda"),
            )
        else:
            with pytest.raises(
                bi.BatchInvariantUnsupportedError, match="not a part of the mode"
            ):
                fp("mm_fp4", N=4096, K=4096, out_dtype=torch.bfloat16)
    with pytest.raises(TypeError):
        fp("mm_fp4", N=4096, K=4096)
    with pytest.raises(TypeError):
        fp("mm_fp4", N=4096, K=4096, out_dtype=torch.bfloat16, dtype=E4)
    with pytest.raises(ValueError):
        fp("mm_fp4", N=4096, K=4096, out_dtype=torch.float32, arch="sm_100")
    with pytest.raises(ValueError):
        fp("mm_fp4", N=4096, K=4096, out_dtype=torch.bfloat16, arch="sm_90")


def test_chunk_rule_takes_no_m_and_counts_whole_granules():
    """A chunk is a whole number of granules: an MMA step that is also a whole scale-factor atom column (64 elements for the
    NVFP4 64-wide step: four 16-element blocks; 128 for the others)."""
    import inspect

    params = list(inspect.signature(chunks_for).parameters)
    assert params == ["N", "K", "in_fmt"] and "M" not in params
    assert chunk_unit("nvfp4") == 64
    for fmt in ("nvfp4_k128", "mxfp4", "mxfp4_k128"):
        assert chunk_unit(fmt) == 128
    for fmt in FP4_FMTS:
        assert chunks_for(4096, 4096, fmt) == 1 and chunks_for(128, 4096, fmt) == 2
        assert chunks_for(64, 8192, fmt) == 8 and chunks_for(128, 8192, fmt) == 4
        assert chunks_for(64, 8256, fmt) == 1
        for N, K in [(64, 64), (512, 65536), (2048, 2048), (8, 1040), (64, 262144)]:
            C = chunks_for(N, K, fmt)
            u = chunk_unit(fmt)
            assert C & (C - 1) == 0 and (C == 1 or (K % (u * C) == 0 and K // C >= 256))
    # where the 64-granule rule cuts 16640 (260 steps) into 4 chunks and 4224 (66 steps) into 2, the 128-granule rules stop
    # at 2 and 1
    assert chunks_for(64, 16640, "nvfp4") == 4 and chunks_for(64, 4224, "nvfp4") == 2
    for fmt in ("nvfp4_k128", "mxfp4", "mxfp4_k128"):
        assert chunks_for(64, 16640, fmt) == 2 and chunks_for(64, 4224, fmt) == 1
    # the SM107 NVFP4 class rule: the 128-wide step's class for the weight shapes it leaves unsplit, else the 64-wide one
    for N, K, cls in [
        (4096, 4096, "nvfp4_k128"),
        (16384, 16384, "nvfp4_k128"),
        (128, 8192, "nvfp4"),
        (512, 16384, "nvfp4"),
        (64, 4096, "nvfp4"),
    ]:
        assert fp4_class_for(N, K, "sm_107") == cls, (N, K)
        assert fp4_class_for(N, K, "sm_100") == "nvfp4"
        assert fp4_class_for(N, K, "sm_103") == "nvfp4"
    assert (
        mx_class_for("mxfp4", "sm_107") == "mxfp4_k128"
        and mx_class_for("mxfp4", "sm_100") == "mxfp4"
        and mx_class_for("mxfp4", "sm_103") == "mxfp4"
    )


def test_block_scaled_spaces():
    """The 4-bit knob products: MMA M 128 / 256 (the 128-row scale-factor atom), tile N 64 .. 256 or a swapped token tile
    (8 / 16 / 32, and 64 for the UE8M0 formats), clusters of at most 4 per dim, the in-cluster split on (1, 1) clusters,
    the B-reuse 512-row tile on the SM107 classes only; the kernel takes K-major weights only; the reference space holds the
    dense records, a decode shape adds the swapped records, a split shape the in-cluster split and the in-CTA chunk-fold
    twins."""
    sizes = {
        "nvfp4": (384, 600, 288),
        "nvfp4_k128": (480, 768, 384),
        "mxfp4": (424, 640, 288),
        "mxfp4_k128": (520, 808, 384),
    }
    for fmt, (plain, with_cc, ref_n) in sizes.items():
        assert len(knob_product(in_fmt=fmt)) == plain, fmt
        assert len(knob_product(in_fmt=fmt, cta_chunks=True)) == with_cc, fmt
        swap_tiles = (8, 16, 32) if fmt in NVFP4_FMTS else (8, 16, 32, 64)
        for k in knob_product(in_fmt=fmt, cta_chunks=True):
            assert k.well_formed(fmt)
            assert not (k.swap_ab or k.cluster_k > 1 or k.b_reuse) or not k.well_formed(
                "e4m3"
            )
            assert k.inst_m in (128, 256) and k.cluster_m <= 4 and k.cluster_n <= 4
            assert k.tile_n in (swap_tiles if k.swap_ab else (64, 128, 192, 256))
            assert k.cluster_k == 1 or (
                k.cluster_shape_mn == (1, 1) and not k.use_2cta_instrs and not k.swap_ab
            )
            assert not k.cta_chunks or (
                k.cluster_k == 1 and not k.swap_ab and not k.b_reuse
            )
            assert not k.b_reuse or (
                fmt.endswith("_k128") and k.use_2cta_instrs and k.tile_m == 512
            )
        assert enumerate_space(*REFERENCE_SHAPE, b_major="n", in_fmt=fmt) == []
        ref = enumerate_space(*REFERENCE_SHAPE, b_major="k", in_fmt=fmt)
        assert len(ref) == ref_n
        assert not any(k.swap_ab or k.cluster_k > 1 or k.cta_chunks for k in ref)
        dec = enumerate_space(8, 4096, 4096, b_major="k", in_fmt=fmt)
        assert set(ref) < set(dec) and all(
            k.swap_ab and k.tile_n >= 8 for k in set(dec) - set(ref)
        )
        wide = set(enumerate_space(64, 4096, 4096, b_major="k", in_fmt=fmt)) - set(ref)
        assert wide == {k for k in dec if k.swap_ab and k.tile_n == 64}
        assert bool(wide) == (fmt in MXFP4_FMTS)  # no 64-wide swapped tile for NVFP4
        c4 = enumerate_space(16, 128, 8192, chunks=4, b_major="k", in_fmt=fmt)
        assert (
            any(k.cluster_k == 4 for k in c4)
            and any(k.cta_chunks for k in c4)
            and not any(k.swap_ab for k in c4)
        )
        assert all(k.tile_n <= 64 for k in c4 if k.cluster_k == 4)
        c2 = enumerate_space(16, 128, 4096, chunks=2, b_major="k", in_fmt=fmt)
        assert all(k.tile_n <= 128 for k in c2 if k.cluster_k == 2) and any(
            k.cluster_k == 2 for k in c2
        )


@pytest.mark.parametrize("family,sm_count", PARTS)
@both_formats
def test_dispatcher_picks_are_members_of_the_space(family, sm_count, nv):
    """R3 membership: every configuration the dispatcher or the tuner's candidate list returns for the part's class of the
    weight shape is a well-formed record the kernel accepts; the candidates are distinct and the default is their first."""
    for M in (1, 8, 16, 64, 300, 4096):
        for N, K in [
            (4096, 4096),
            (14336, 4096),
            (128, 4096),
            (64, 8192),
            (2048, 16384),
            (128, 8192),
        ]:
            fmt = fp4_class_for(N, K, family) if nv else mx_class_for("mxfp4", family)
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


def test_flag_bypasses_the_operator_backend_selection():
    """Under ``batch_invariant=True`` the operator's auto-backend heuristic yields a non-empty list whatever the backends'
    availability: the flag path runs the class, so a missing FP4 backend never turns into a backend error."""
    from flashinfer.gemm.gemm_base import _heuristic_func_mm_fp4

    a = torch.zeros(4, 64, dtype=torch.uint8)
    b = torch.zeros(128, 64, dtype=torch.uint8).t()
    sf = torch.zeros(1024, dtype=torch.uint8)
    assert _heuristic_func_mm_fp4([], a, b, sf, sf, batch_invariant=True)
    assert _heuristic_func_mm_fp4(
        [], a, b, sf, sf, use_nvfp4=False, batch_invariant=True
    )


def test_unsupported_device_raises_before_any_kernel():
    a = torch.zeros(4, 64, dtype=torch.uint8)
    w = torch.zeros(128, 64, dtype=torch.uint8).t()
    sf = torch.zeros(1024, dtype=torch.uint8)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="gemm.fp4"):
        direct(a, w, sf, sf)


# ----------------------------------------------------------------------------------------------- on the device


def _backend_outputs(a, b, sfa, sfb, alpha, nv, out_dtype=torch.bfloat16):
    """FlashInfer's own backends on the same operands, those that run the shape on this device (not trtllm: its weight is
    shuffled, another input)."""
    outs = {}
    for be in ("cute-dsl", "cutlass", "cudnn"):
        try:
            outs[be] = flashinfer.mm_fp4(
                a,
                b,
                sfa,
                sfb,
                alpha,
                out_dtype,
                None,
                block_size=block(nv),
                backend=be,
                use_nvfp4=nv,
            )
        except Exception:  # noqa: BLE001  (the backend does not run this shape / device / format)
            continue
    return outs


@needs_device
@both_formats
@pytest.mark.parametrize(
    "M,N,K,od",
    [
        (1000, 4096, 4096, torch.bfloat16),
        (512, 4000, 1024, torch.float16),
        (300, 128, 4096, torch.bfloat16),
    ],
)
def test_matches_flashinfer_backends(M, N, K, od, nv):
    """Accuracy yardstick: the max norm-relative error against the exact product of the dequantised operands is of the size
    of the FP4 backends' (CUTLASS / cuDNN / CuTe DSL through ``mm_fp4``), for the unsplit class and the split class
    (N = 128: C = 2); and bit equality with FlashInfer's CuTe-DSL FP4 kernel where it runs the shape and the class is the
    device's unsplit one (the same instruction, K ascending; a power-of-two alpha, exact wherever the kernel applies it)."""
    span = None if od == torch.bfloat16 else (1 if nv else 2)
    a, sfa, ra = adv_fp4(M, K, 11, nv, span=span)
    b, sfb, rb = weight(N, K, 12, nv, span=span)
    alpha = alpha_of(2.0**-3)
    ours = bi_mm(a, b, sfa, sfb, alpha, nv, out_dtype=od)
    assert torch.isfinite(ours.float()).all()
    err = norm_rel_err(ours, a, ra, b.t(), rb, alpha, nv)
    refs = _backend_outputs(a, b, sfa, sfb, alpha, nv, od)
    errs = {be: norm_rel_err(c, a, ra, b.t(), rb, alpha, nv) for be, c in refs.items()}
    assert errs, "no FP4 backend ran the shape"
    assert err <= 1.1 * min(errs.values()) + 1e-6, (err, errs)
    fmt = fp4_format_for(torch.device("cuda"), nv, N, K)
    if "cute-dsl" in refs and chunks_for(N, K, fmt) == 1 and not fmt.endswith("_k128"):
        # the unsplit 64-wide-step class only: FlashInfer's kernel folds the whole K in one pass of that instruction
        assert torch.equal(bits(ours), bits(refs["cute-dsl"])), (
            "the CuTe-DSL kernel's bits differ"
        )


@needs_device
@both_formats
@pytest.mark.parametrize("M,N,K", [(300, 4096, 4096), (37, 128, 8192)])
def test_alpha_chain_is_one_mul_rn_then_one_cvt_rn(M, N, K, nv):
    """A single packed byte in each activation row makes the unscaled dot exact. Check alpha and final rounding."""
    a, sfa, a_rows = adv_fp4(M, K, 15, nv, span=1)
    b, sfb, b_rows = weight(N, K, 16, nv, span=1)
    a[:, 1:] = 0
    f32 = (
        (dequant(a, a_rows, nv)[:, :2] @ dequant(b.t(), b_rows, nv)[:, :2].t())
        .float()
        .to(a.device)
    )
    for al in (0.3, 1.0 / 3.0, 2.0**-7):
        alpha = alpha_of(al)
        for od in (torch.bfloat16, torch.float16):
            chain = (f32 * alpha).to(od)
            assert torch.equal(
                bits(bi_mm(a, b, sfa, sfb, alpha, nv, out_dtype=od)), bits(chain)
            ), (al, od)


@needs_device
@both_formats
@pytest.mark.parametrize("K", [96, 4128])
def test_partial_last_step_reads_finite_scale_padding(K, nv):
    """K a multiple of 32 but not of the class's step: the last step is partial, its values TMA zero fill and its scales
    the scale buffers' padding bytes (zero here, as the quantizers write them), so the fold equals the exact product; the
    swizzled buffers of the 128x4 layout hold those padding blocks."""
    M, N = 37, 256
    a, sfa, ra = adv_fp4(M, K, 141, nv, span=2)
    b, sfb, rb = weight(N, K, 142, nv, span=2)
    alpha = alpha_of(0.5)
    out = bi_mm(a, b, sfa, sfb, alpha, nv)
    assert torch.isfinite(out.float()).all()
    assert norm_rel_err(out, a, ra, b.t(), rb, alpha, nv) <= 2.0**-8


@needs_device
@both_formats
@pytest.mark.parametrize("M,N,K", [(300, 4096, 4096), (37, 128, 8192), (8, 4096, 4096)])
def test_determinism_and_graph_replay(M, N, K, nv):
    a, sfa, a_rows = adv_fp4(M, K, 21, nv)
    b, sfb, b_rows = weight(N, K, 22, nv)
    alpha = alpha_of(0.7)
    ref = bits(bi_mm(a, b, sfa, sfb, alpha, nv)).clone()
    for _ in range(3):
        assert torch.equal(bits(bi_mm(a, b, sfa, sfb, alpha, nv)), ref)
    out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        bi_mm(
            a, b, sfa, sfb, alpha, nv, out=out
        )  # warm-up on the side stream: the compile happens outside the capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            bi_mm(a, b, sfa, sfb, alpha, nv, out=out)
    torch.cuda.current_stream().wait_stream(s)
    for _ in range(2):
        out.zero_()
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(bits(out), ref)
    # Leave two active K elements so the unscaled dot product has an exact independent oracle.
    a_unit = a.clone()
    a_unit[:, 1:] = 0
    lhs = dequant(a_unit, a_rows, nv)[:, :2]
    rhs = dequant(b.t(), b_rows, nv)[:, :2]
    expected = bits((lhs @ rhs.t()).to(torch.bfloat16)).to(a.device)
    # Capture the implicit unit scale, then check eager execution before the first replay.
    out1 = torch.empty_like(out)
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        g1 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g1):
            bi_mm(a_unit, b, sfa, sfb, None, nv, out=out1)
    torch.cuda.current_stream().wait_stream(s)
    assert torch.equal(bits(bi_mm(a_unit, b, sfa, sfb, None, nv)), expected)
    out1.zero_()
    g1.replay()
    torch.cuda.synchronize()
    assert torch.equal(bits(out1), expected)


@needs_device
@both_formats
def test_batch_invariance_along_rows(nv):
    """R2: a row's bits do not depend on the number of rows (the dispatcher's M-dependent picks included: the swapped decode
    records at small M, the dense tiles above), its position or the other rows; the parts' picks agree."""
    N, K = 4096, 4096
    a, sfa, ra = adv_fp4(2049, K, 31, nv)
    b, sfb, _ = weight(N, K, 32, nv)
    alpha = alpha_of(0.9)
    full = bits(bi_mm(a, b, sfa, sfb, alpha, nv))
    for M in (1, 2, 7, 16, 32, 37, 63, 64, 65, 128, 256, 512, 1000, 2049):
        sub = a[:M].contiguous()
        assert torch.equal(
            bits(bi_mm(sub, b, swizzle_sf(ra[:M]), sfb, alpha, nv)), full[:M]
        ), f"M={M}"
    # a row in another batch: the row's scales travel with it
    x5, _, r5 = adv_fp4(5, K, 33, nv)
    x7, _, r7 = adv_fp4(7, K, 34, nv)
    x = torch.cat([x5, a[100:101], x7])
    sfx = swizzle_sf(torch.cat([r5, ra[100:101], r7]))
    assert torch.equal(bits(bi_mm(x, b, sfx, sfb, alpha, nv))[5], full[100])


@needs_device
@both_formats
@pytest.mark.parametrize("N,K", [(2048, 4096), (128, 8192)])
def test_permutation_and_subbatch_invariance(N, K, nv):
    """R2 in both forms along the rows, for the unsplit (N = 2048) and the split (N = 128, C = 4) class."""
    M = 512
    a, sfa, ra = adv_fp4(M, K, 41, nv)
    b, sfb, _ = weight(N, K, 42, nv)
    alpha = alpha_of(1.1)
    full = bits(bi_mm(a, b, sfa, sfb, alpha, nv))
    perm = torch.randperm(M, generator=torch.Generator().manual_seed(1))
    assert torch.equal(
        bits(
            bi_mm(a[perm.cuda()].contiguous(), b, swizzle_sf(ra[perm]), sfb, alpha, nv)
        ),
        full[perm.cuda()],
    )
    for k in (1, 2, 4, 8, 16, 37):
        idx = torch.randperm(M, generator=torch.Generator().manual_seed(k))[:k]
        got = bits(
            bi_mm(a[idx.cuda()].contiguous(), b, swizzle_sf(ra[idx]), sfb, alpha, nv)
        )
        assert torch.equal(got, full[idx.cuda()]), f"k={k}"


@needs_device
@both_formats
@pytest.mark.parametrize("N,K", [(4096, 4096), (128, 8192)])
def test_public_rows_keep_their_bits_inside_autotune(N, K, nv):
    """The public batch-invariant entry agrees before, during and after autotuning."""
    a, sfa, ra = adv_fp4(300, K, 101, nv)
    b, sfb, _ = weight(N, K, 102, nv)
    alpha = alpha_of(0.55)

    def run(M):
        return bits(bi_mm(a[:M].contiguous(), b, swizzle_sf(ra[:M]), sfb, alpha, nv))

    ref = [run(M) for M in (1, 16, 300)]
    with autotune(True):
        tuned = [run(M) for M in (1, 16, 300)]
    post = [run(M) for M in (1, 16, 300)]
    for r, t, p in zip(ref, tuned, post, strict=True):
        assert torch.equal(r, t) and torch.equal(r, p)
    assert torch.equal(post[1], post[2][:16])


@needs_device
def test_public_api_skips_backend_selection(monkeypatch):
    """batch_invariant=True uses the cuDNN plan directly, so the @backend_requirement wrapper must not run the auto selection
    for it (host cost only): a selecting call rewrites `mm_fp4.suitable_auto_backends`, a bypassed one leaves it."""
    a, sfa, _ = adv_fp4(16, 4096, 131, True)
    b, sfb, _ = weight(4096, 4096, 132, True)
    sentinel = ["sentinel"]
    # the wrapper creates the attribute at its first auto selection, so it may not exist yet
    monkeypatch.setattr(
        flashinfer.mm_fp4, "suitable_auto_backends", sentinel, raising=False
    )
    bi_mm(a, b, sfa, sfb, alpha_of(0.5), True)
    assert flashinfer.mm_fp4.suitable_auto_backends is sentinel


@needs_device
@pytest.mark.parametrize("M,N,K", [(16, 4096, 4096), (37, 128, 8192)])
def test_public_output_buffer_and_graph_replay(M, N, K):
    """A caller-provided output survives graph replay and must be disjoint from the inputs."""
    nv = True
    a, sfa, a_rows = adv_fp4(M, K, 181, nv)
    b, sfb, b_rows = weight(N, K, 182, nv)
    alpha = alpha_of(0.5)
    got = bi_mm(a, b, sfa, sfb, alpha, nv)
    assert norm_rel_err(got, a, a_rows, b.t(), b_rows, alpha, nv) <= 2**-8
    ref = bits(got)
    out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    assert bi_mm(a, b, sfa, sfb, alpha, nv, out=out) is out
    assert torch.equal(bits(out), ref)
    chunks = chunks_for(N, K, "nvfp4")
    if chunks > 1:
        storage = torch.full(
            (chunks * M * N + 8,), float("nan"), dtype=torch.float32, device=a.device
        )
        scratch = storage[4:-4]
        out.fill_(float("nan"))
        with patch.object(
            gemm_base,
            "_execute_cudnn_batch_invariant",
            wraps=gemm_base._execute_cudnn_batch_invariant,
        ) as execute:
            assert (
                mm_fp4_batch_invariant(
                    a, b, sfa, sfb, alpha, out=out, workspace=scratch
                )
                is out
            )
        used = execute.call_args.args[0].get_workspace_size() // scratch.element_size()
        assert torch.equal(bits(out), ref)
        assert torch.isnan(storage[:4]).all() and torch.isnan(storage[4 + used :]).all()
        if used:
            assert torch.isfinite(scratch[:used]).any()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        bi_mm(a, b, sfa, sfb, alpha, nv, out=out)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            bi_mm(a, b, sfa, sfb, alpha, nv, out=out)
    torch.cuda.current_stream().wait_stream(s)
    out.zero_()
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(bits(out), ref)
    buf = torch.empty(M * max(2 * N, K // 2 + 1), dtype=torch.uint8, device="cuda")
    alias_out = buf[: 2 * M * N].view(torch.bfloat16).view(M, N)
    for pitch in (K // 2, K // 2 + 1):
        alias_a = buf.as_strided((M, K // 2), (pitch, 1))
        with pytest.raises(
            bi.BatchInvariantUnsupportedError, match="A overlapping the output"
        ):
            bi_mm(alias_a, b, sfa, sfb, alpha, nv, out=alias_out)


@needs_device
def test_unsupported_inputs_raise_and_nothing_falls_back():
    nv = True
    a, sfa, ra = adv_fp4(16, 4096, 121, nv)
    b, sfb, _ = weight(4096, 4096, 122, nv)
    alpha = alpha_of(0.5)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match='backend="auto"'):
        flashinfer.mm_fp4(
            a, b, sfa, sfb, alpha, backend="cutlass", batch_invariant=True
        )
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="8x4"):
        bi_mm(a, b, sfa, sfb, alpha, nv, use_8x4_sf_layout=True)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="K-major"):
        bi_mm(a, b.contiguous(), sfa, sfb, alpha, nv)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="packed fp4"):
        direct(a.view(torch.int8), b, sfa, sfb, alpha, nv)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="uint8"):
        direct(a, b, sfa.view(torch.int8), sfb, alpha, nv)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="uint8"):
        # e4m3-typed scales are NVFP4's; for MXFP4 they would be read as UE8M0 exponents, so they are a typed error there
        mm_fp4_batch_invariant(
            a, b, sfa.view(E4), sfb.view(E4), alpha, use_nvfp4=False, block_size=32
        )
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="at least"):
        direct(a, b, sfa, sfb[:-512], alpha, nv)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="on A's device"):
        bi_mm(a, b, sfa, sfb.cpu(), alpha, nv)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="multiple of 32"):
        direct(a[:, :2040], b[:2040], sfa, sfb, alpha, nv)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="2-D operands"):
        direct(a.unsqueeze(0), b, sfa, sfb, alpha, nv)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="one scalar alpha"):
        direct(a, b, sfa, sfb, torch.full((16,), 0.5, device="cuda"), nv)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="float32 tensor alpha"):
        direct(a, b, sfa, sfb, alpha.half(), nv)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="float32 tensor alpha"):
        direct(a, b, sfa, sfb, 0.5, nv)
    # the public API hands batch-invariant calls to the class without running the backend checks: the class's typed
    # error (a ValueError, as before)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="bf16 or fp16 output"):
        bi_mm(a, b, sfa, sfb, alpha, nv, out_dtype=torch.float32)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="out"):
        direct(
            a,
            b,
            sfa,
            sfb,
            alpha,
            nv,
            out=torch.empty(16, 4096, dtype=torch.float16, device="cuda"),
        )
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="block_size"):
        mm_fp4_batch_invariant(a, b, sfa, sfb, alpha, use_nvfp4=nv, block_size=32)
    # the quantizers' dtypes are the same bytes: e4m3-typed NVFP4 scales and the packed-fp4 torch dtype
    ref = bits(bi_mm(a, b, sfa, sfb, alpha, nv))
    assert torch.equal(bits(bi_mm(a, b, sfa.view(E4), sfb.view(E4), alpha, nv)), ref)
    if FP4X2 is not None:
        assert torch.equal(
            bits(bi_mm(a.view(FP4X2), b.view(FP4X2), sfa, sfb, alpha, nv)), ref
        )
    # nothing to launch at M = 0
    out = bi_mm(
        a[:0], b, swizzle_sf(torch.zeros(0, 256, dtype=torch.uint8)), sfb, alpha, nv
    )
    assert out.shape == (0, 4096)
