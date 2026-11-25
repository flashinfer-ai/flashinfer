# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Batch-invariant MXFP8 GEMM: scale layout, dequantized reference, row invariance and CUDA Graph replay."""

from unittest.mock import patch

import pytest
import torch

import flashinfer
from flashinfer import batch_invariant as bi
from flashinfer.gemm import gemm_base
from flashinfer.autotuner import autotune
from flashinfer.gemm.batch_invariant.api import (
    mm_mxfp8_batch_invariant,
    mx_format_for,
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
    mx_class_for,
    rank_topk,
)
from flashinfer.gemm.batch_invariant.spec import MXFP8_FMTS, chunk_unit

E4 = torch.float8_e4m3fn
needs_device = pytest.mark.usefixtures("cudnn_batch_invariant_device")


def bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16) if t.element_size() == 2 else t.view(torch.int32)


def swizzle_sf(sf_rows: torch.Tensor) -> torch.Tensor:
    """The (rows, K / 32) uint8 block scales in the 128x4 swizzled layout ``mxfp8_quantize(..., is_sf_swizzled_layout=True)``
    writes: atoms of 128 rows x 4 blocks (512 bytes; row r, block j of an atom at byte (r % 32) * 16 + (r // 32) * 4 + j),
    K-atoms consecutive, row-atoms strided by the K-atoms of the whole K; the padding bytes are zero."""
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


def adv_mx(rows: int, K: int, seed: int, cancel: bool = True, span: int = 12):
    """An adversarial MXFP8 operand: e4m3 values over the format's range with random signs, near-cancelling neighbour pairs
    and zeros of both signs; UE8M0 block scales spread over 2^-12 .. 2^12 so that the products of neighbouring blocks differ
    by orders of magnitude and any accumulation-order difference reaches the output bits. Returns the e4m3 (rows, K) tensor
    and its scales in the 128x4 swizzled layout. ``span``: the scales' exponent range (a narrower one keeps an fp16 output
    finite over a long K)."""
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


def weight(N: int, K: int, seed: int, span: int = 12):
    """``mm_mxfp8``'s weight: the (K, N) transpose of a row-major (N, K) e4m3 tensor and its scales."""
    w, sf = adv_mx(N, K, seed, cancel=False, span=span)
    return w.t(), sf


def bi_mm(a, b, sfa, sfb, **kw):
    return flashinfer.mm_mxfp8(
        a, b, sfa, sfb, backend="auto", batch_invariant=True, **kw
    )


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


def dequant(q: torch.Tensor, sf_flat: torch.Tensor) -> torch.Tensor:
    rows, K = q.shape
    return q.double().cpu() * e8m0(sf_flat, rows, K).repeat_interleave(32, dim=1)


def norm_rel_err(c, a, sfa, bt, sfb) -> float:
    """The max norm-relative error against the exact product of the dequantised operands."""
    A, B = dequant(a, sfa), dequant(bt, sfb)
    ref = A @ B.t()
    scale = A.abs() @ B.abs().t()
    return ((c.double().cpu() - ref).abs() / (scale + 1e-30)).max().item()


# ----------------------------------------------------------------------------------------------- device-free


def test_fingerprint_is_a_function_of_the_weight_shape_the_output_format_and_the_architecture():
    f107 = bi.fingerprint(
        "mm_mxfp8", N=4096, K=4096, out_dtype=torch.bfloat16, arch="sm_107"
    )
    f100 = bi.fingerprint(
        "mm_mxfp8", N=4096, K=4096, out_dtype=torch.bfloat16, arch="sm_100"
    )
    assert f107.startswith("gemm-mxfp8-k64-kfold-") and f100.startswith(
        "gemm-mxfp8-kfold-"
    )
    assert f107 != f100  # two instructions, two classes
    assert f100 == bi.fingerprint(
        "mm_mxfp8", N=4096, K=4096, out_dtype="bf16", arch="sm_103"
    )
    assert f107 == bi.fingerprint(
        "mm_mxfp8", N=4096, K=4096, out_dtype="bf16", arch=(10, 7)
    )
    assert bi.fingerprint(
        "mm_mxfp8", N=128, K=8192, out_dtype=torch.float16, arch="sm_100"
    ).startswith("gemm-mxfp8-kchunk4-")
    # the output dtype and the raw K are dimensions of the specification; M and the layout of the scale buffers are not
    distinct = {
        bi.fingerprint("mm_mxfp8", N=4096, K=K, out_dtype=od, arch=arch)
        for K in (4096, 4096 + 32)
        for od in (torch.bfloat16, torch.float16)
        for arch in ("sm_100", "sm_107")
    }
    assert len(distinct) == 8
    assert f100 != bi.fingerprint(
        "bmm_fp8", N=4096, K=4096, dtype=E4, out_dtype=torch.bfloat16
    )
    assert bi.class_name("mm_mxfp8") == "gemm.mxfp8"
    if torch.cuda.is_available():
        # the default architecture is the current device's: on a part of the mode its fingerprint; on any other part the
        # typed error, because the family selects the block-scaled instruction and no other part's class is borrowed
        if torch.cuda.get_device_capability() in FAMILY_OF_CC:
            assert bi.fingerprint(
                "mm_mxfp8", N=4096, K=4096, out_dtype=torch.bfloat16
            ) == bi.fingerprint(
                "mm_mxfp8",
                N=4096,
                K=4096,
                out_dtype=torch.bfloat16,
                arch=torch.device("cuda"),
            )
        else:
            with pytest.raises(
                bi.BatchInvariantUnsupportedError, match="not a part of the mode"
            ):
                bi.fingerprint("mm_mxfp8", N=4096, K=4096, out_dtype=torch.bfloat16)
    with pytest.raises(TypeError):
        bi.fingerprint("mm_mxfp8", N=4096, K=4096)
    with pytest.raises(TypeError):
        bi.fingerprint("mm_mxfp8", N=4096, K=4096, out_dtype=torch.bfloat16, dtype=E4)
    with pytest.raises(ValueError):
        bi.fingerprint(
            "mm_mxfp8", N=4096, K=4096, out_dtype=torch.float32, arch="sm_100"
        )
    with pytest.raises(ValueError):
        bi.fingerprint(
            "mm_mxfp8", N=4096, K=4096, out_dtype=torch.bfloat16, arch="sm_90"
        )


def test_chunk_rule_takes_no_m_and_counts_128_element_granules():
    import inspect

    params = list(inspect.signature(chunks_for).parameters)
    assert params == ["N", "K", "in_fmt"] and "M" not in params
    for fmt in MXFP8_FMTS:
        assert (
            chunk_unit(fmt) == 128
        )  # four 32-element block scales: one scale-factor atom column, a whole number of steps
        assert chunks_for(4096, 4096, fmt) == 1 and chunks_for(128, 4096, fmt) == 2
        assert chunks_for(64, 8192, fmt) == 8 and chunks_for(128, 8192, fmt) == 4
        # a chunk is a whole number of granules: where the fp8 rule splits 16640 into 8 chunks (520 x 32), the MX rule stops at 2
        assert chunks_for(64, 16640, "e4m3") == 8 and chunks_for(64, 16640, fmt) == 2
        assert chunks_for(64, 8256, fmt) == 1 and chunks_for(64, 4224, fmt) == 1
        for N, K in [
            (64, 64),
            (512, 65536),
            (2048, 2048),
            (64, 8208),
            (8, 1040),
            (64, 262144),
        ]:
            C = chunks_for(N, K, fmt)
            assert C & (C - 1) == 0 and (
                C == 1 or (K % (128 * C) == 0 and K // C >= 256)
            )
    assert chunks_for(4096, 4096, "mxfp8") == chunks_for(4096, 4096, "mxfp8_k64")


def test_block_scaled_spaces():
    """The block-scaled knob products: MMA M 128 / 256 (the 128-row scale-factor atom), tile N 64 .. 256 or a swapped token
    tile 8 / 16 / 32 / 64, clusters of at most 4 per dim, the in-cluster split on (1, 1) clusters, the B-reuse 512-row tile on
    the SM107 class only; the kernel takes K-major weights only; the reference space holds the dense records, a decode shape
    adds the swapped records, a split shape the in-cluster split and the in-CTA chunk-fold twins."""
    assert (
        len(knob_product(in_fmt="mxfp8")) == 424
        and len(knob_product(in_fmt="mxfp8_k64")) == 520
    )
    assert len(knob_product(in_fmt="mxfp8", cta_chunks=True)) == 640
    assert len(knob_product(in_fmt="mxfp8_k64", cta_chunks=True)) == 808
    for fmt in MXFP8_FMTS:
        for k in knob_product(in_fmt=fmt, cta_chunks=True):
            assert k.well_formed(fmt)
            assert not (k.swap_ab or k.cluster_k > 1 or k.b_reuse) or not k.well_formed(
                "e4m3"
            )
            assert k.inst_m in (128, 256) and k.cluster_m <= 4 and k.cluster_n <= 4
            assert k.tile_n in ((8, 16, 32, 64) if k.swap_ab else (64, 128, 192, 256))
            assert k.cluster_k == 1 or (
                k.cluster_shape_mn == (1, 1) and not k.use_2cta_instrs and not k.swap_ab
            )
            assert not k.cta_chunks or (
                k.cluster_k == 1 and not k.swap_ab and not k.b_reuse
            )
            assert not k.b_reuse or (
                fmt == "mxfp8_k64" and k.use_2cta_instrs and k.tile_m == 512
            )
        assert enumerate_space(*REFERENCE_SHAPE, b_major="n", in_fmt=fmt) == []
        ref = enumerate_space(*REFERENCE_SHAPE, b_major="k", in_fmt=fmt)
        assert not any(k.swap_ab or k.cluster_k > 1 or k.cta_chunks for k in ref)
        dec = enumerate_space(8, 4096, 4096, b_major="k", in_fmt=fmt)
        assert set(ref) < set(dec) and all(
            k.swap_ab and k.tile_n >= 8 for k in set(dec) - set(ref)
        )
        assert set(enumerate_space(64, 4096, 4096, b_major="k", in_fmt=fmt)) - set(
            ref
        ) == {k for k in dec if k.swap_ab and k.tile_n == 64}
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
    assert len(enumerate_space(*REFERENCE_SHAPE, b_major="k", in_fmt="mxfp8")) == 288
    assert (
        len(enumerate_space(*REFERENCE_SHAPE, b_major="k", in_fmt="mxfp8_k64")) == 384
    )
    assert (
        mx_class_for("mxfp8", "sm_107") == "mxfp8_k64"
        and mx_class_for("mxfp8", "sm_100") == "mxfp8"
    )
    assert mx_class_for("mxfp8", "sm_103") == "mxfp8"


def test_in_cluster_split_is_not_a_placement_beyond_the_grid_z_limit():
    """The in-cluster split launches one cluster per output tile on grid.z, which the hardware caps at 65535: above that
    tile count the record is not accepted for the shape and the dispatcher's pick is the workspace split (the same class,
    the same bits), with no M-dependence in the class itself."""
    from flashinfer.gemm.batch_invariant.blocks.operands import MAX_GRID_Z

    N, K = 512, 16384
    for fmt in MXFP8_FMTS:
        C = chunks_for(N, K, fmt)
        assert C == 2
        ck = next(
            k
            for k in enumerate_space(64, N, K, chunks=C, b_major="k", in_fmt=fmt)
            if k.cluster_k == C
        )
        small = (
            MAX_GRID_Z // -(-N // ck.tile_n)
        ) * ck.tile_m  # the largest M whose tile count fits
        assert can_implement(ck, small, N, K, "k", C, fmt)
        assert not can_implement(ck, small + ck.tile_m, N, K, "k", C, fmt)
        for M in (2_000_000, 2_200_000):
            for family, sm in PARTS:
                if mx_class_for("mxfp8", family) != fmt:
                    continue
                pick = default_config(M, N, K, C, "k", sm, family, fmt)
                assert can_implement(pick, M, N, K, "k", C, fmt), (M, pick.describe())
                tiles = -(-M // pick.tile_m) * -(-N // pick.tile_n)
                assert pick.cluster_k == 1 or tiles <= MAX_GRID_Z, (M, pick.describe())


@pytest.mark.parametrize("family,sm_count", PARTS)
def test_dispatcher_picks_are_members_of_the_space(family, sm_count):
    """R3 membership: every configuration the dispatcher or the tuner's candidate list returns for the part's MXFP8 class is
    a well-formed record the kernel accepts for the shape; the candidates are distinct and the default is their first."""
    fmt = mx_class_for("mxfp8", family)
    for M in (1, 8, 16, 64, 300, 4096):
        for N, K in [
            (4096, 4096),
            (14336, 4096),
            (128, 4096),
            (64, 8192),
            (2048, 16384),
            (128, 8192),
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
    w = torch.zeros(128, 128).to(E4).t()
    sf = torch.zeros(512, dtype=torch.uint8)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="gemm.mxfp8"):
        mm_mxfp8_batch_invariant(a, w, sf, sf)


# ----------------------------------------------------------------------------------------------- on the device


def _backend_outputs(a, b, sfa, sfb, out_dtype=torch.bfloat16):
    """FlashInfer's own backends on the same operands, those that run the shape on this device."""
    outs = {}
    for be in ("cute-dsl", "cutlass", "cudnn", "trtllm"):
        try:
            outs[be] = flashinfer.mm_mxfp8(
                a, b, sfa, sfb, out_dtype=out_dtype, backend=be
            )
        except Exception:  # noqa: BLE001  (the backend does not run this shape / device)
            continue
    return outs


@needs_device
@pytest.mark.parametrize(
    "M,N,K,od",
    [
        (1000, 4096, 4096, torch.bfloat16),
        (512, 4000, 1024, torch.float16),
        (300, 128, 4096, torch.bfloat16),
    ],
)
def test_matches_flashinfer_backends(M, N, K, od):
    """Accuracy yardstick: the max norm-relative error against the exact product of the dequantised operands is of the size
    of the MXFP8 backends' (CUTLASS / cuDNN / CuTe DSL through ``mm_mxfp8``), for the unsplit class and the split class
    (N = 128: C = 2); and bit equality with FlashInfer's CuTe-DSL MXFP8 kernel where it runs the shape (the same
    instruction, K ascending, one conversion: the same per-element dataflow)."""
    a, sfa = adv_mx(M, K, 11, cancel=False)
    b, sfb = weight(N, K, 12)
    ours = bi_mm(a, b, sfa, sfb, out_dtype=od)
    err = norm_rel_err(ours, a, sfa, b.t(), sfb)
    refs = _backend_outputs(a, b, sfa, sfb, od)
    errs = {be: norm_rel_err(c, a, sfa, b.t(), sfb) for be, c in refs.items()}
    assert errs, "no MXFP8 backend ran the shape"
    assert err <= 1.1 * min(errs.values()) + 1e-6, (err, errs)
    if (
        "cute-dsl" in refs
        and chunks_for(N, K, mx_format_for(torch.device("cuda"))) == 1
    ):
        # the unsplit class only: FlashInfer's kernel folds the whole K in one pass, the split class folds chunks
        assert torch.equal(bits(ours), bits(refs["cute-dsl"])), (
            "the CuTe-DSL kernel's bits differ"
        )


@needs_device
def test_partial_last_step_reads_finite_scale_padding():
    """K a multiple of 32 but not of the SM107 class's 64-wide step: the last step is partial, its values TMA zero fill and
    its scales the scale buffers' padding bytes (zero here, as the quantizer writes them), so the fold equals the exact
    product (on the 32-wide step's class every step is whole)."""
    M, N, K = 37, 256, 4128
    a, sfa = adv_mx(M, K, 141, span=3)
    b, sfb = weight(N, K, 142, span=3)
    out = bi_mm(a, b, sfa, sfb)
    assert torch.isfinite(out.float()).all()
    assert norm_rel_err(out, a, sfa, b.t(), sfb) <= 2.0**-8


def test_flag_bypasses_the_operator_backend_selection():
    """Under ``batch_invariant=True`` the operator's auto-backend heuristic yields a non-empty list whatever the backends'
    availability: the flag path runs the class, so a missing MXFP8 backend never turns into a backend error."""
    from flashinfer.gemm.gemm_base import _heuristic_func_mm_mxfp8

    a = torch.zeros(4, 128).to(E4)
    b = torch.zeros(128, 128).to(E4).t()
    sf = torch.zeros(512, dtype=torch.uint8)
    assert _heuristic_func_mm_mxfp8([], a, b, sf, sf, batch_invariant=True)


def test_unlisted_compute_capability_raises():
    """The family selects the block-scaled instruction, hence the bits: a capability outside the mode's parts raises instead
    of borrowing a family."""
    from flashinfer.gemm.batch_invariant.heuristic import family_of

    assert family_of((10, 7)) == "sm_107" and family_of((10, 0)) == "sm_100"
    with pytest.raises(
        bi.BatchInvariantUnsupportedError, match="not a part of the mode"
    ):
        family_of((9, 0))
    with pytest.raises(
        bi.BatchInvariantUnsupportedError, match="not a part of the mode"
    ):
        bi.fingerprint(
            "mm_mxfp8", N=4096, K=4096, out_dtype=torch.bfloat16, arch=(12, 0)
        )


@needs_device
@pytest.mark.parametrize("M,N,K", [(300, 4096, 4096), (37, 128, 8192), (8, 4096, 4096)])
def test_determinism_and_graph_replay(M, N, K):
    a, sfa = adv_mx(M, K, 21)
    b, sfb = weight(N, K, 22)
    ref = bits(bi_mm(a, b, sfa, sfb)).clone()
    for _ in range(3):
        assert torch.equal(bits(bi_mm(a, b, sfa, sfb)), ref)
    out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        bi_mm(
            a, b, sfa, sfb, out=out
        )  # warm-up on the side stream: the compile happens outside the capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            bi_mm(a, b, sfa, sfb, out=out)
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
    a, sfa = adv_mx(2049, K, 31)
    b, sfb = weight(N, K, 32)
    full = bits(bi_mm(a, b, sfa, sfb))
    for M in (1, 2, 7, 16, 32, 37, 63, 64, 65, 128, 256, 512, 1000, 2049):
        sub = a[:M].contiguous()
        rows = e8m0(sfa, 2049, K)[:M]
        sfm = swizzle_sf((torch.log2(rows) + 127).to(torch.uint8))
        assert torch.equal(bits(bi_mm(sub, b, sfm, sfb)), full[:M]), f"M={M}"
    # a row in another batch: the row's scales travel with it
    x = torch.cat([adv_mx(5, K, 33)[0], a[100:101], adv_mx(7, K, 34)[0]])
    sfx = swizzle_sf(
        torch.cat(
            [
                (torch.log2(e8m0(adv_mx(5, K, 33)[1], 5, K)) + 127),
                (torch.log2(e8m0(sfa, 2049, K)[100:101]) + 127),
                (torch.log2(e8m0(adv_mx(7, K, 34)[1], 7, K)) + 127),
            ]
        ).to(torch.uint8)
    )
    assert torch.equal(bits(bi_mm(x, b, sfx, sfb))[5], full[100])


@needs_device
@pytest.mark.parametrize("N,K", [(2048, 4096), (128, 8192)])
def test_permutation_and_subbatch_invariance(N, K):
    """R2 in both forms along the rows, for the unsplit (N = 2048) and the split (N = 128, C = 4) class."""
    M = 512
    a, sfa = adv_mx(M, K, 41)
    b, sfb = weight(N, K, 42)
    full = bits(bi_mm(a, b, sfa, sfb))
    rows = (torch.log2(e8m0(sfa, M, K)) + 127).to(torch.uint8)
    perm = torch.randperm(M, generator=torch.Generator().manual_seed(1))
    assert torch.equal(
        bits(bi_mm(a[perm.cuda()].contiguous(), b, swizzle_sf(rows[perm]), sfb)),
        full[perm.cuda()],
    )
    for k in (1, 2, 4, 8, 16, 37):
        idx = torch.randperm(M, generator=torch.Generator().manual_seed(k))[:k]
        got = bits(bi_mm(a[idx.cuda()].contiguous(), b, swizzle_sf(rows[idx]), sfb))
        assert torch.equal(got, full[idx.cuda()]), f"k={k}"


@needs_device
@pytest.mark.parametrize("N,K", [(4096, 4096), (128, 8192)])
def test_public_rows_keep_their_bits_inside_autotune(N, K):
    """The public batch-invariant entry agrees before, during and after autotuning."""
    a, sfa = adv_mx(300, K, 101)
    b, sfb = weight(N, K, 102)
    rows = (torch.log2(e8m0(sfa, 300, K)) + 127).to(torch.uint8)

    def run(M):
        return bits(bi_mm(a[:M].contiguous(), b, swizzle_sf(rows[:M]), sfb))

    ref = [run(M) for M in (1, 16, 300)]
    with autotune(True):
        tuned = [run(M) for M in (1, 16, 300)]
    post = [run(M) for M in (1, 16, 300)]
    for r, t, p in zip(ref, tuned, post, strict=True):
        assert torch.equal(r, t) and torch.equal(r, p)
    assert torch.equal(post[1], post[2][:16])


@needs_device
def test_public_api_skips_backend_selection(monkeypatch):
    """Batch-invariant calls bypass the ordinary backend heuristic."""
    from flashinfer.gemm import gemm_base

    def forbidden_heuristic(*args, **kwargs):
        raise AssertionError("batch-invariant call reached backend selection")

    monkeypatch.setattr(gemm_base, "_heuristic_func_mm_mxfp8", forbidden_heuristic)
    a, sfa = adv_mx(16, 4096, 131)
    b, sfb = weight(4096, 4096, 132)
    bi_mm(a, b, sfa, sfb)


@needs_device
@pytest.mark.parametrize("M,N,K", [(16, 4096, 4096), (37, 128, 8192)])
def test_public_output_buffer_and_graph_replay(M, N, K):
    """A caller-provided output survives graph replay and must be disjoint from the inputs."""
    a, sfa = adv_mx(M, K, 181)
    b, sfb = weight(N, K, 182)
    got = bi_mm(a, b, sfa, sfb)
    assert norm_rel_err(got, a, sfa, b.t(), sfb) <= 2**-8
    ref = bits(got)
    out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    assert bi_mm(a, b, sfa, sfb, out=out) is out and torch.equal(bits(out), ref)
    chunks = chunks_for(N, K, "mxfp8")
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
                mm_mxfp8_batch_invariant(a, b, sfa, sfb, out=out, workspace=scratch)
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
        bi_mm(a, b, sfa, sfb, out=out)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            bi_mm(a, b, sfa, sfb, out=out)
    torch.cuda.current_stream().wait_stream(s)
    out.zero_()
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(bits(out), ref)
    buf = torch.empty(M * max(2 * N, K), dtype=torch.uint8, device="cuda")
    alias_a = buf[: M * K].view(E4).view(M, K)
    alias_out = buf[: 2 * M * N].view(torch.bfloat16).view(M, N)
    with pytest.raises(
        bi.BatchInvariantUnsupportedError, match="A overlapping the output"
    ):
        bi_mm(alias_a, b, sfa, sfb, out=alias_out)


@needs_device
def test_unsupported_inputs_raise_and_nothing_falls_back():
    a, sfa = adv_mx(16, 4096, 121)
    b, sfb = weight(4096, 4096, 122)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match='backend="auto"'):
        flashinfer.mm_mxfp8(a, b, sfa, sfb, backend="cutlass", batch_invariant=True)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="8x4"):
        bi_mm(a, b, sfa, sfb, use_8x4_sf_layout=True)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="K-major"):
        bi_mm(a, b.contiguous(), sfa, sfb)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="float8_e4m3fn"):
        mm_mxfp8_batch_invariant(a.view(torch.uint8), b, sfa, sfb)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="uint8"):
        mm_mxfp8_batch_invariant(a, b, sfa.view(torch.int8), sfb)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="at least"):
        mm_mxfp8_batch_invariant(a, b, sfa, sfb[:-512])
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="on A's device"):
        bi_mm(a, b, sfa, sfb.cpu())
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="multiple of 32"):
        mm_mxfp8_batch_invariant(a[:, :4080], b[:4080], sfa, sfb)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="2-D operands"):
        mm_mxfp8_batch_invariant(a.unsqueeze(0), b, sfa, sfb)
    # the public API hands batch-invariant calls to the class without running the backend checks: the class's typed
    # error (a ValueError, as before)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="bf16 or fp16 output"):
        bi_mm(a, b, sfa, sfb, out_dtype=torch.float32)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="out"):
        mm_mxfp8_batch_invariant(
            a,
            b,
            sfa,
            sfb,
            out=torch.empty(16, 4096, dtype=torch.float16, device="cuda"),
        )
    out = bi_mm(a[:0], b, swizzle_sf(torch.zeros(0, 128, dtype=torch.uint8)), sfb)
    assert out.shape == (0, 4096)
