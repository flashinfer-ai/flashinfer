# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The batch-invariant 4-bit GEMMs: ``flashinfer.mm_fp4(a, b, a_descale, b_descale, alpha, backend="auto",
use_nvfp4=..., batch_invariant=True)``.

NVFP4 (one e4m3 block scale per 16 elements): the classes ``gemm.nvfp4.kfold`` / ``kchunk[C]`` (the SM100 / SM103
instruction: 64 e2m1 elements per step) and, on SM107, ``gemm.nvfp4.kfold128`` for the weight shapes the chunk rule leaves
unsplit (the SM107 instruction: 128 per step). MXFP4 (one UE8M0 scale per 32): ``gemm.mxfp4.kfold`` / ``kchunk[C]`` and on
SM107 ``gemm.mxfp4.kfold128`` / ``kchunk128[C]``. All are the dense K fold with a block-scaled step, the scales applied
inside the instruction, and the chain ``cvt.rn(mul.rn(fold, alpha))`` with the caller's scalar alpha. There is no exact CPU
oracle for the step, so the classes are checked by bit agreement across their configurations -- the dense records, the
swapped decode records, the in-cluster split, the in-CTA chunk fold, the workspace split and (SM107) the B-reuse record,
the hardware hypothesis of ``lean/BIKernels/GEMMConfig.lean`` -- by the launch-model properties R1-R2 on adversarial
operands and scales, by the exactness of the alpha chain against the exposed fp32 fold, by bit equality with FlashInfer's
own CuTe-DSL FP4 kernel (the same instruction, the same K order) and by an accuracy yardstick against the other backends.
Device-free tests cover the fingerprints (``use_nvfp4`` and the architecture are dimensions; the SM107 NVFP4 class rule),
the chunk rule, the spaces, the dispatcher and the tuner's tactic encoding; the GPU tests need SM100 / SM103 / SM107.
"""

import dataclasses

import pytest
import torch

import flashinfer
from flashinfer import batch_invariant as bi
from flashinfer.autotuner import AutoTuner, autotune
from flashinfer.gemm.batch_invariant import tuning
from flashinfer.gemm.batch_invariant.api import (
    MM_FP4_ARCHS,
    fp4_format_for,
    mm_fp4_batch_invariant,
    part_of,
)
from flashinfer.gemm.batch_invariant.blocks.operands import can_implement
from flashinfer.gemm.batch_invariant.generator import (
    REFERENCE_SHAPE,
    Knobs,
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
_supported = torch.cuda.is_available() and bi.is_supported_device(
    torch.device("cuda"), MM_FP4_ARCHS
)
needs_device = pytest.mark.skipif(
    not _supported,
    reason="needs a CUDA device of SM100 / SM103 / SM107 with the CuTe DSL",
)
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


def unswizzle_sf(flat: torch.Tensor, rows: int, kb: int) -> torch.Tensor:
    rt, kt, off = _sf_offsets(rows, kb)
    return flat.cpu()[off.reshape(-1)].view(rt * 128, kt * 4)[:rows, :kb].clone()


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


def test_tactic_encoding_round_trips():
    for fmt, shape in (
        ("nvfp4", (16, 128, 8192, 4)),
        ("nvfp4_k128", (8, 4096, 4096, 1)),
        ("mxfp4", (300, 4096, 4096, 1)),
        ("mxfp4_k128", (16, 128, 4096, 2)),
    ):
        M, N, K, C = shape
        for cfg in enumerate_space(M, N, K, chunks=C, b_major="k", in_fmt=fmt)[::37]:
            t = tuning.knobs_to_tactic(cfg)
            assert (
                len(t) == 12
                and tuning.knobs_from_tactic(t) == cfg
                and tuning.knobs_from_tactic(list(t)) == cfg
            )
    assert tuning.OUT_INDEX["mm_fp4"] == 5 and set(tuning.TUNING_CONFIG_MM_FP4) == {
        "nvfp4",
        "mxfp4",
    }


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


def test_runner_shape_matches_the_dispatcher_for_every_api():
    """The autotuner's runner recovers the real (M, N, K) from its inputs for every API (K in elements: the 4-bit operands
    are byte-packed, the others are not), so its default tactic is the dispatcher's own pick at the real shape."""
    M, N, K = 300, 4096, 4096
    a16 = torch.zeros(M, K, dtype=torch.bfloat16)
    b16 = torch.zeros(K, N, dtype=torch.bfloat16)
    a8, b8 = a16.to(torch.float8_e4m3fn), b16.to(torch.float8_e4m3fn)
    a4 = torch.zeros(M, K // 2, dtype=torch.uint8)
    b4 = torch.zeros(K // 2, N, dtype=torch.uint8)
    sf = torch.zeros(1, dtype=torch.uint8)
    cases = [
        ("mm_bf16", "bf16", "n", [a16, b16, None, None]),
        ("bmm_fp8", "e4m3", "k", [a8[None], b8[None], None, None, None]),
        ("mm_mxfp8", "mxfp8", "k", [a8, b8, sf, sf, None]),
        ("mm_fp4", "nvfp4", "k", [a4, b4, sf, sf, None, None]),
        ("mm_fp4", "mxfp4", "k", [a4, b4, sf, sf, None, None]),
    ]
    for api, fmt, lay, inputs in cases:
        runner = tuning.BatchInvariantGemmRunner(api, fmt, lay, 1, 148, "sm_100")
        assert runner._shape(inputs) == (M, N, K), api
        assert runner.config_for(inputs, -1) == default_config(
            M, N, K, 1, lay, 148, "sm_100", fmt
        ), api


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
    """The chain: the fp32 fold (exposed by the direct entry at alpha = 1), one ``mul.rn`` by alpha and one conversion; the
    same in torch for a bf16 and an fp16 output and an alpha that is not a power of two; None is alpha 1.0; the swapped
    decode record and the split placements apply it in the same place."""
    a, sfa, _ = adv_fp4(M, K, 15, nv, span=1 if nv else 2)
    b, sfb, _ = weight(N, K, 16, nv, span=1 if nv else 2)
    f32 = direct(a, b, sfa, sfb, alpha_of(1.0), nv, out_dtype=torch.float32)
    assert torch.equal(
        bits(f32), bits(direct(a, b, sfa, sfb, None, nv, out_dtype=torch.float32))
    )
    for al in (0.3, 1.0 / 3.0, 2.0**-7):
        alpha = alpha_of(al)
        for od in (torch.bfloat16, torch.float16):
            chain = (f32 * alpha).to(od)
            assert torch.equal(
                bits(bi_mm(a, b, sfa, sfb, alpha, nv, out_dtype=od)), bits(chain)
            ), (al, od)
    alpha = alpha_of(0.3)
    chain = (f32 * alpha).to(torch.bfloat16)
    fmt = fp4_format_for(torch.device("cuda"), nv, N, K)
    C = chunks_for(N, K, fmt)
    picks = [
        k
        for k in enumerate_space(M, N, K, chunks=C, b_major="k", in_fmt=fmt)
        if (k.swap_ab or k.cluster_k == C or k.cta_chunks)
        and can_implement(k, M, N, K, "k", C, fmt)
    ]
    assert picks
    for k in picks[::5] + [picks[-1]]:
        assert torch.equal(
            bits(direct(a, b, sfa, sfb, alpha, nv, config=k, chunks=C)), bits(chain)
        ), k.describe()


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
    f32 = direct(a, b, sfa, sfb, alpha, nv, out_dtype=torch.float32)
    assert torch.equal(bits(f32.to(torch.bfloat16)), bits(out))


@needs_device
@both_formats
@pytest.mark.parametrize("M,N,K", [(300, 4096, 4096), (37, 128, 8192), (8, 4096, 4096)])
def test_determinism_and_graph_replay(M, N, K, nv):
    from flashinfer.gemm.batch_invariant import api

    a, sfa, _ = adv_fp4(M, K, 21, nv)
    b, sfb, _ = weight(N, K, 22, nv)
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
    # alpha = None (1.0) captured first: the ones-tensor made under capture is not the cached one (its fill would only
    # run on replay), so an eager call before the first replay is right, and the replay is too
    ref1 = bits(bi_mm(a, b, sfa, sfb, alpha_of(1.0), nv)).clone()
    saved = dict(api._ONES)
    api._ONES.clear()
    try:
        out1 = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            g1 = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g1):
                bi_mm(a, b, sfa, sfb, None, nv, out=out1)
        torch.cuda.current_stream().wait_stream(s)
        assert torch.equal(
            bits(bi_mm(a, b, sfa, sfb, None, nv)), ref1
        )  # eager, before any replay
        out1.zero_()
        g1.replay()
        torch.cuda.synchronize()
        assert torch.equal(bits(out1), ref1)
    finally:
        api._ONES.clear()
        api._ONES.update(saved)


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
    # the dispatcher's picks for other row counts and parts compute the same bits
    fmt = fp4_format_for(torch.device("cuda"), nv, N, K)
    for M in (16, 1000):
        for cfg in (
            default_config(M, N, K, 1, "k", 204, "sm_107", fmt),
            default_config(2049, N, K, 1, "k", 148, "sm_100", fmt),
            default_config(1, N, K, 1, "k", 148, "sm_100", fmt),
        ):
            if can_implement(cfg, M, N, K, "k", 1, fmt):
                got = bits(
                    direct(
                        a[:M].clone(), b, swizzle_sf(ra[:M]), sfb, alpha, nv, config=cfg
                    )
                )
                assert torch.equal(got, full[:M]), (M, cfg.describe())


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
@pytest.mark.parametrize("M,N,K", [(1000, 4096, 4096), (37, 2048, 4000 + 96)])
def test_config_agreement(M, N, K, nv):
    """R3: every configuration the tuner may choose (the dispatcher's top candidates on every part that runs the class) and
    a spread of hand-picked records (1- and 2-CTA MMA, the 64- / 128- / 192- / 256-wide tiles, clusters, both rasters,
    swizzles, the B-reuse tile on SM107) give identical bits."""
    fmt = fp4_format_for(torch.device("cuda"), nv, N, K)
    a, sfa, _ = adv_fp4(M, K, 61, nv)
    b, sfb, _ = weight(N, K, 62, nv)
    alpha = alpha_of(0.6)
    cfgs = []
    for family, sm in PARTS:
        if (
            fp4_class_for(N, K, family) if nv else mx_class_for("mxfp4", family)
        ) == fmt:
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
        got = bits(direct(a, b, sfa, sfb, alpha, nv, config=c))
        if ref is None:
            ref = got
        assert torch.equal(got, ref), f"{c.describe()} differs"
        checked += 1
    assert checked >= 14


@needs_device
@both_formats
def test_swapped_decode_records_compute_the_class(nv):
    """The decode specialisation (``swap_ab``: the weights as the kernel's A, the tokens as its B, one token tile) on the
    narrow 8 / 16 / 32 tiles (and the 64-wide one for MXFP4; the NVFP4 spaces have no swapped record above 32 rows), 1- and
    2-CTA, clusters along M, every raster: the same bits as the dense records (the block-scaled step is symmetric in its
    operands), at M = 1 .. 64 and for a bf16 and an fp16 output."""
    N, K = 4096, 4096
    fmt = fp4_format_for(torch.device("cuda"), nv, N, K)
    for M, od in (
        (1, torch.bfloat16),
        (5, torch.float16),
        (8, torch.bfloat16),
        (16, torch.float16),
        (33, torch.bfloat16),
        (64, torch.bfloat16),
    ):
        span = None if od == torch.bfloat16 else (1 if nv else 2)
        a, sfa, _ = adv_fp4(M, K, 70 + M, nv, span=span)
        b, sfb, _ = weight(N, K, 71, nv, span=span)
        alpha = alpha_of(0.5 if od == torch.bfloat16 else 2.0**-6)
        dense = Knobs((128, 128), (1, 1), False, True, "m", 1)
        ref = bits(direct(a, b, sfa, sfb, alpha, nv, out_dtype=od, config=dense))
        recs = [
            k for k in enumerate_space(M, N, K, b_major="k", in_fmt=fmt) if k.swap_ab
        ]
        pick = {(k.tile_n, k.tile_m, k.cluster_m, k.raster_along): k for k in recs}
        if nv and M > 32:
            # no 64-wide token tile in the NVFP4 spaces: above 32 rows the decode shape runs the dense records
            assert not recs
            assert torch.equal(
                bits(bi_mm(a, b, sfa, sfb, alpha, nv, out_dtype=od)), ref
            )
            continue
        assert len(pick) >= 4, M
        wide = [k for k in recs if k.tile_n == 64][:4]
        assert bool(wide) == (not nv)  # the 64-wide token tile: the UE8M0 formats only
        for k in list(pick.values())[::3] + wide:
            got = bits(direct(a, b, sfa, sfb, alpha, nv, out_dtype=od, config=k))
            assert torch.equal(got, ref), (M, k.describe())
        assert torch.equal(
            bits(bi_mm(a, b, sfa, sfb, alpha, nv, out_dtype=od)), ref
        )  # the dispatcher's own pick


@needs_device
@both_formats
@pytest.mark.parametrize(
    "M,N,K,od",
    [
        (37, 128, 8192, torch.bfloat16),
        (300, 128, 16384, torch.float16),
        (16, 128, 4096, torch.bfloat16),
    ],
)
def test_split_implementations_agree(M, N, K, od, nv):
    """The split class ``kchunk[C]`` has three implementations: the workspace split (chunks on different CTAs, an fp32
    workspace, a fixup fold in chunk order and the chain), the in-CTA chunk fold (one CTA folds every chunk of its tile
    through a TMEM running sum) and the in-cluster split (the chunks on the CTAs of one cluster, the fold through
    distributed shared memory). They are one term in Lean (``ctaSplitBS_eq_kchunkBS``, ``clusterSplitMX4_eq_kchunkMX4`` and
    the SM107 twins) and one bit pattern here; and the split class is a different specification from the unsplit fold."""
    fmt = fp4_format_for(torch.device("cuda"), nv, N, K)
    C = chunks_for(N, K, fmt)
    assert C > 1
    span = None if od == torch.bfloat16 else (1 if nv else 2)
    a, sfa, _ = adv_fp4(M, K, 81, nv, span=span)
    b, sfb, _ = weight(N, K, 82, nv, span=span)
    alpha = alpha_of(0.8 if od == torch.bfloat16 else 2.0**-7)
    ws = Knobs((128, 64), (1, 1), False, True, "m", 1)
    ref = bits(direct(a, b, sfa, sfb, alpha, nv, out_dtype=od, config=ws, chunks=C))
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
    assert C not in (2, 4) or any(k.cluster_k > 1 for k in recs)
    for k in recs:
        got = bits(direct(a, b, sfa, sfb, alpha, nv, out_dtype=od, config=k, chunks=C))
        assert torch.equal(got, ref), k.describe()
    assert torch.equal(bits(bi_mm(a, b, sfa, sfb, alpha, nv, out_dtype=od)), ref)
    if od == torch.bfloat16:
        # a different specification from the unsplit fold: the fp32 folds differ wherever the fold rounds (the fp16 cases run
        # narrow scales for a finite output, under which a 4-bit fold can be exact in fp32 and the association cannot show)
        f32 = dict(out_dtype=torch.float32, config=ws)
        assert not torch.equal(
            bits(direct(a, b, sfa, sfb, alpha, nv, chunks=1, **f32)),
            bits(direct(a, b, sfa, sfb, alpha, nv, chunks=C, **f32)),
        )


@needs_device
@both_formats
def test_persistent_grid_size_does_not_change_bits(nv):
    """R2 for the SM count: a smaller persistent grid is what a smaller part, an SM carveout, MPS or a green context
    produce; it only decides which CTA computes which tile (and, for the split class, which chunk)."""
    for M, N, K in [(1000, 4096, 4096), (64, 64, 65536)]:
        fmt = fp4_format_for(torch.device("cuda"), nv, N, K)
        a, sfa, _ = adv_fp4(M, K, 91, nv)
        b, sfb, _ = weight(N, K, 92, nv)
        alpha = alpha_of(0.4)
        C = chunks_for(N, K, fmt)
        base = default_config(M, N, K, C, "k", 204, "sm_107", fmt)
        ref = bits(direct(a, b, sfa, sfb, alpha, nv, config=base, chunks=C))
        for g in (1, 3, 17, 74):
            cfg = dataclasses.replace(base, grid_clusters=g)
            assert torch.equal(
                bits(direct(a, b, sfa, sfb, alpha, nv, config=cfg, chunks=C)), ref
            ), f"grid={g} at {(M, N, K)}"


@needs_device
@both_formats
def test_the_two_instructions_are_two_classes(nv):
    """On SM107 both instructions exist: the sm_100 classes (``in_fmt="nvfp4"`` / ``"mxfp4"``) run there too and are
    different specifications from the 128-wide step's; the NVFP4 class rule picks the 64-wide step's class for a split weight
    shape (the same bits as on SM100 / SM103)."""
    fam = part_of(torch.device("cuda"))[1]
    base, k128 = ("nvfp4", "nvfp4_k128") if nv else ("mxfp4", "mxfp4_k128")
    a, sfa, _ = adv_fp4(300, 4096, 93, nv)
    b, sfb, _ = weight(2048, 4096, 94, nv)
    alpha = alpha_of(0.75)
    if fam != "sm_107":
        with pytest.raises(bi.BatchInvariantUnsupportedError, match="on SM107"):
            direct(a, b, sfa, sfb, alpha, nv, in_fmt=k128)
        assert fp4_format_for(torch.device("cuda"), nv, 2048, 4096) == base
        assert fp4_format_for(torch.device("cuda"), nv, 128, 8192) == base
        return
    assert fp4_format_for(torch.device("cuda"), nv, 2048, 4096) == k128
    wide = bits(direct(a, b, sfa, sfb, alpha, nv))
    narrow = bits(direct(a, b, sfa, sfb, alpha, nv, in_fmt=base))
    assert not torch.equal(wide, narrow)
    for fmt, ref in ((base, narrow), (k128, wide)):
        for c in rank_topk(300, 2048, 4096, 1, "k", 204, 4, "sm_107", fmt):
            assert torch.equal(
                bits(direct(a, b, sfa, sfb, alpha, nv, config=c, in_fmt=fmt)), ref
            ), (
                fmt,
                c.describe(),
            )
    # a split weight shape: NVFP4 runs the 64-wide step's class on SM107 (the rule), MXFP4 the 128-wide one
    a2, sfa2, _ = adv_fp4(300, 8192, 95, nv)
    b2, sfb2, _ = weight(128, 8192, 96, nv)
    dev_fmt = fp4_format_for(torch.device("cuda"), nv, 128, 8192)
    assert dev_fmt == (base if nv else k128)
    other = k128 if nv else base
    assert not torch.equal(
        bits(direct(a2, b2, sfa2, sfb2, alpha, nv)),
        bits(direct(a2, b2, sfa2, sfb2, alpha, nv, in_fmt=other)),
    )
    assert torch.equal(
        bits(bi_mm(a2, b2, sfa2, sfb2, alpha, nv)),
        bits(direct(a2, b2, sfa2, sfb2, alpha, nv, in_fmt=dev_fmt)),
    )


@needs_device
@both_formats
@pytest.mark.parametrize("N,K", [(4096, 4096), (128, 8192)])
def test_autotuning_profiles_every_candidate_and_keeps_the_bits(N, K, nv):
    """Under ``autotune(True)`` the FlashInfer autotuner profiles the class's candidates (none may be dropped as
    unsupported; the profiled row counts get scale buffers of their own size) and the tuned process computes the same
    bits as the untuned one, during and after tuning."""
    a, sfa, ra = adv_fp4(300, K, 101, nv)
    b, sfb, _ = weight(N, K, 102, nv)
    alpha = alpha_of(0.55)

    def run(M):
        return bits(bi_mm(a[:M].contiguous(), b, swizzle_sf(ra[:M]), sfb, alpha, nv))

    ref = [run(M) for M in (1, 16, 300)]
    tuner = AutoTuner.get()
    before = {
        k: len(v)
        for k, v in tuner.stats.failed_tactics.items()
        if tuning.CUSTOM_OP_MM_FP4 in k
    }
    with autotune(True):
        tuned = [run(M) for M in (1, 16, 300)]
    after = {
        k: len(v)
        for k, v in tuner.stats.failed_tactics.items()
        if tuning.CUSTOM_OP_MM_FP4 in k
    }
    assert after == before, f"tactics failed while profiling: {after}"
    post = [run(M) for M in (1, 16, 300)]
    for r, t, p in zip(ref, tuned, post, strict=True):
        assert torch.equal(r, t) and torch.equal(r, p)
    assert torch.equal(post[1], post[2][:16])


@needs_device
def test_public_api_skips_backend_selection(monkeypatch):
    """batch_invariant=True never runs a backend, so the @backend_requirement wrapper must not run the auto selection
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
def test_unsupported_inputs_raise_and_nothing_falls_back():
    nv = True
    a, sfa, ra = adv_fp4(16, 4096, 121, nv)
    b, sfb, _ = weight(4096, 4096, 122, nv)
    alpha = alpha_of(0.5)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match='backend="auto"'):
        flashinfer.mm_fp4(
            a, b, sfa, sfb, alpha, backend="cutlass", batch_invariant=True
        )
    # the class's checks, reached through the API (which runs no backend checks for the flag) and through its entries
    auto = tuning.mm_fp4_batch_invariant_auto
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="8x4"):
        auto(a, b, sfa, sfb, alpha, torch.bfloat16, None, 16, True, "auto", True)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="block_size"):
        auto(a, b, sfa, sfb, alpha, torch.bfloat16, None, 32, False, "auto", True)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="bf16 or fp16"):
        auto(a, b, sfa, sfb, alpha, torch.float32, None, 16, False, "auto", True)
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
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="chunks = 0"):
        direct(a, b, sfa, sfb, alpha, nv, chunks=0)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="granules"):
        direct(a, b, sfa, sfb, alpha, nv, chunks=128)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="enumerated space"):
        direct(
            a,
            b,
            sfa,
            sfb,
            alpha,
            nv,
            config=Knobs((128, 64), (1, 1), False, True, "m", 1, 0, False, 1, True),
            chunks=1,
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


@needs_device
@both_formats
def test_chunk_count_and_workspace_checks(nv):
    """A caller workspace may not overlap the output or alpha: the split class writes its partials there before the fixup
    reads them; the workspace split at M beyond the hardware's grid.y limit strides over rows and agrees with its own
    sub-batch."""
    M, N, K = 64, 128, 8192
    a, sfa, _ = adv_fp4(M, K, 131, nv)
    b, sfb, _ = weight(N, K, 132, nv)
    alpha = alpha_of(0.45)
    fmt = fp4_format_for(torch.device("cuda"), nv, N, K)
    C = chunks_for(N, K, fmt)
    assert C > 1
    ws = torch.empty(C * M * N + 64, dtype=torch.float32, device="cuda")
    ref = bits(bi_mm(a, b, sfa, sfb, alpha, nv)).clone()
    wsplit = Knobs((128, 64), (1, 1), False, True, "m", 1)
    assert torch.equal(
        bits(
            direct(a, b, sfa, sfb, alpha, nv, config=wsplit, workspace=ws[: C * M * N])
        ),
        ref,
    )
    out = ws.view(torch.bfloat16)[: M * N].view(M, N)
    with pytest.raises(
        bi.BatchInvariantUnsupportedError, match="overlapping the workspace"
    ):
        direct(a, b, sfa, sfb, alpha, nv, out=out, config=wsplit, workspace=ws)
    with pytest.raises(
        bi.BatchInvariantUnsupportedError, match="overlapping the workspace"
    ):
        direct(a, b, sfa, sfb, ws[:1], nv, config=wsplit, workspace=ws)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="A overlapping"):
        direct(
            ws.view(torch.uint8)[: 16 * K // 2].view(16, K // 2),
            b,
            swizzle_sf(torch.zeros(16, K // block(nv), dtype=torch.uint8)),
            sfb,
            alpha,
            nv,
            config=wsplit,
            workspace=ws,
        )
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="scales overlapping"):
        direct(
            a,
            b,
            sfa,
            ws.view(torch.uint8)[: sfb.numel()],
            alpha,
            nv,
            config=wsplit,
            workspace=ws,
        )
    big_m = 70000
    ab, sfab, rab = adv_fp4(big_m, K, 133, nv)
    full = bits(direct(ab, b, sfab, sfb, alpha, nv, config=wsplit, chunks=C))
    assert torch.equal(
        bits(bi_mm(ab[:777].contiguous(), b, swizzle_sf(rab[:777]), sfb, alpha, nv)),
        full[:777],
    )
