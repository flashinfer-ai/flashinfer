# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The batch-invariant block-scale fp8 GEMM: ``flashinfer.gemm_fp8_nt_groupwise(a, b, a_scale, b_scale, scale_major_mode,
backend="auto", batch_invariant=True)``, the class ``gemm.e4m3.kblock128``.

Per output element the class folds every 128-element K block with the 32-wide ``kind::f8f6f4`` step from +0 (the block
partial ``P_kb``) and the block partials in block order from +0 with ``fma.rn(mul.rn(a_scale[m, kb], b_scale[n // 128, kb]),
P_kb, acc)``, then converts once. The opaque part of the class is the block partial, which IS the per-tensor fp8 class at
K = 128 with unit scales and an fp32 output (``x * 1.0`` is exact): ``block_partials`` computes it with
``mm_fp8_batch_invariant`` block by block on the device. Everything after it is correctly rounded fp32 arithmetic the CPU
reproduces bit for bit (``exact_promotion``), so GPU == exact model is a bitwise assertion here; then the launch-model
properties R1 / R2 (the scales travelling with their rows), bit agreement across the enumerated configuration space (the
hardware hypothesis for the fp8 instruction variants and the promotion's independence of the tile), both scale layouts,
dispatcher membership, the persistent grid, the autotuner, Trace Apply, FlashInfer's own groupwise backends as a tolerance
yardstick (their association is not the class's) and the error contract. Device-free tests cover the fingerprint, the
space, the dispatcher and the tuner's tactic encoding; the GPU tests need SM100 / SM103 / SM107.
"""

import contextlib
import dataclasses

import pytest
import torch

import flashinfer
from flashinfer import batch_invariant as bi
from flashinfer.autotuner import AutoTuner, autotune
from flashinfer.batch_invariant.fp_interface import exact
from flashinfer.gemm.batch_invariant import tuning
from flashinfer.gemm.batch_invariant.api import (
    GEMM_FP8BS_ARCHS,
    gemm_fp8_nt_groupwise_batch_invariant,
    mm_fp8_batch_invariant,
    native_k_staging,
)
from flashinfer.gemm.batch_invariant.blocks.operands import can_implement
from flashinfer.gemm.batch_invariant.generator import (
    Knobs,
    enumerate_space,
    knob_product,
)
from flashinfer.gemm.batch_invariant.heuristic import PARTS, capacity_for
from flashinfer.gemm.batch_invariant.schedule import (
    chunks_for,
    default_config,
    rank_topk,
)
from flashinfer.gemm.batch_invariant.spec import (
    BLOCK_K,
    SCALE_BLOCK_N,
    GemmSpec,
    spec_fingerprint,
)

E4, E5 = torch.float8_e4m3fn, torch.float8_e5m2
FMT = "e4m3_bs"
_supported = torch.cuda.is_available() and bi.is_supported_device(
    torch.device("cuda"), GEMM_FP8BS_ARCHS
)
needs_device = pytest.mark.skipif(
    not _supported,
    reason="needs a CUDA device of SM100 / SM103 / SM107 with the CuTe DSL",
)

#: The DeepSeek-V3 dense weight shapes (N, K) and the row counts the dispatcher is checked at.
DS_PAIRS = (
    (7168, 7168),
    (7168, 2048),
    (2048, 7168),
    (4096, 7168),
    (18432, 7168),
    (7168, 18432),
    (576, 7168),
    (7168, 16384),
)
DS_MS = (1, 16, 128, 1024, 4096, 16384)
#: Odd M, a partial last tile, N = 576 / 136 (a partial last weight scale block), the 7168 square at decode.
EXACT_SHAPES = [
    (37, 256, 512),
    (300, 576, 1024),
    (1000, 4096, 2048),
    (16, 7168, 7168),
    (2049, 136, 256),
]
#: K = 128: one block (the instruction variants and one promotion alone); then several blocks at odd shapes.
FULL_SPACE_SHAPES = [
    (64, 4096, 128),
    (200, 8192, 256),
    (37, 2048, 1024),
    (16, 8192, 8192),
]
#: A few rows in the K layout (the K-native staging) with a short last ring stage: K of one, two and three blocks (the short
#: stage alone; a stage holds four blocks) and of six and seven (a whole stage, then the short one); the exact-model shapes at
#: these row counts are whole stages.
NATIVE_TAIL_SHAPES = [
    (64, 4096, 128),
    (37, 2048, 256),
    (16, 576, 384),
    (16, 2048, 768),
    (16, 576, 896),
]
#: Hand-picked records: every tile N, both CTA groups, clusters to 4, both rasters, swizzles to 8.
RECORDS = [
    Knobs((128, 16), (1, 1), False, True, "m", 1),
    Knobs((128, 32), (1, 4), False, True, "m", 1),
    Knobs((128, 64), (1, 2), False, True, "n", 2),
    Knobs((128, 128), (4, 1), False, True, "m", 8),
    Knobs((256, 32), (2, 2), True, True, "n", 4),
    Knobs((256, 64), (2, 1), True, True, "m", 1),
    Knobs((256, 128), (2, 1), True, True, "m", 1),
    Knobs((256, 128), (4, 1), True, True, "n", 2),
]
#: One record of the per-tensor fp8 class the block partials run with (every record computes the same bits, R3), so the exact
#: model does not depend on that class's dispatcher.
PARTIAL_RECORD = Knobs((128, 64), (1, 1), False, True, "m", 1)


def bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16 if t.element_size() == 2 else torch.int32)


def adv_fp8(*shape, seed: int, cancel: bool = True) -> torch.Tensor:
    """Adversarial e4m3 operand: the format's exponent range, random signs, near-cancelling neighbour pairs, zeros of
    both signs, so that any accumulation-order or scaling difference reaches the output bits."""
    g = torch.Generator().manual_seed(seed)
    e = (torch.rand(*shape, generator=g) * 2 - 1) * 7
    x = (torch.rand(*shape, generator=g) * 2 - 1) * torch.exp2(e)
    if cancel and shape[-1] >= 8:
        k = torch.arange(0, shape[-1] - 1, 8)
        x[..., k + 1] = -x[..., k] * (1 + 2.0**-3)
    z = torch.randint(0, 61, shape, generator=g)
    x[z == 0] = 0.0
    x[z == 1] = -0.0
    return x.to(E4).cuda()


def scales(rows: int, KB: int, seed: int, span: int = 6) -> torch.Tensor:
    """fp32 block scales: powers of two over 2^-span .. 2^span times a factor in [1, 1.4) (not powers of two: the fma's
    rounding is live)."""
    g = torch.Generator().manual_seed(seed)
    s = torch.exp2((torch.rand(rows, KB, generator=g) * 2 - 1) * span) * (
        1 + 0.4 * torch.rand(rows, KB, generator=g)
    )
    return s.float().cuda()


def operands(M: int, N: int, K: int, seed: int = 0):
    """``a (M, K)``, the row-major weight ``w (N, K)``, ``sa (M, KB)`` and ``sb (ceil(N / 128), KB)`` in the K layout."""
    a = adv_fp8(M, K, seed=seed + 1)
    w = adv_fp8(N, K, seed=seed + 2, cancel=False)
    KB = K // BLOCK_K
    return a, w, scales(M, KB, seed + 3), scales(-(-N // SCALE_BLOCK_N), KB, seed + 4)


def ours(a, w, sa, sb, **kw):
    return gemm_fp8_nt_groupwise_batch_invariant(a, w, sa, sb, "K", **kw)


def api(a, w, sa, sb, mode: str = "K", **kw):
    if mode == "K":
        return flashinfer.gemm.gemm_fp8_nt_groupwise(
            a,
            w,
            sa,
            sb,
            scale_major_mode="K",
            backend="auto",
            batch_invariant=True,
            **kw,
        )
    return flashinfer.gemm.gemm_fp8_nt_groupwise(
        a,
        w,
        sa.t().contiguous(),
        sb.t().contiguous(),
        scale_major_mode="MN",
        backend="auto",
        batch_invariant=True,
        **kw,
    )


# ----------------------------------------------------------------------------------------------- the exact model


def block_partials(a: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``P[kb, m, n]`` (fp32, on the device): the per-tensor fp8 class's fold of block ``kb`` from +0, through
    ``mm_fp8_batch_invariant`` with unit scales and an fp32 output on the (M, 128) x (128, N) slices, at one pinned record.
    The weight is padded to a multiple of 16 columns with zero rows (every output element is its own fold, so the padding
    changes nothing) and the padding dropped."""
    M, K = a.shape
    N = w.shape[0]
    N16 = -(-N // 16) * 16
    if N16 != N:
        w = torch.cat([w, torch.zeros(N16 - N, K, dtype=w.dtype, device=w.device)], 0)
    one = torch.ones(1, dtype=torch.float32, device=a.device)
    P = torch.empty(K // BLOCK_K, M, N16, dtype=torch.float32, device=a.device)
    for kb in range(K // BLOCK_K):
        a_blk = a[:, kb * BLOCK_K : (kb + 1) * BLOCK_K].contiguous()
        b_blk = w[:, kb * BLOCK_K : (kb + 1) * BLOCK_K].contiguous().t()
        mm_fp8_batch_invariant(
            a_blk,
            b_blk,
            one,
            one,
            out=P[kb],
            out_dtype=torch.float32,
            config=PARTIAL_RECORD,
            chunks=1,
        )
    return P[:, :, :N] if N16 != N else P


def exact_promotion(
    P: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor
) -> torch.Tensor:
    """The class's fp32 fold over the device block partials: ``acc_0 = +0``, ``acc_{kb+1} = fma.rn(mul.rn(sa[m, kb],
    sb[n // 128, kb]), P_kb, acc_kb)``, every operation correctly rounded. The two fp32 factors of a product are exact in
    binary64, so ``mul.rn`` is one binary64 multiply followed by the cast; for the fma, ``round64(exact product + acc)``
    followed by the cast equals the single correctly rounded fp32 result whenever the binary64 sum is exact (TwoSum residue
    0) or is not an fp32 rounding midpoint; the remaining elements (an inexact binary64 sum on a midpoint, or in the fp32
    subnormal range) are evaluated with the rational ``exact.fma``. Returns fp32 (M, N) on the CPU."""
    P, sa, sb = P.cpu(), sa.cpu().float(), sb.cpu().float()
    KB, M, N = P.shape
    col_blk = torch.arange(N) // SCALE_BLOCK_N
    acc = torch.zeros(M, N, dtype=torch.float32)
    tiny = 2 * 2.0**-126
    for kb in range(KB):
        scale = (sa[:, kb].double()[:, None] * sb[col_blk, kb].double()[None, :]).to(
            torch.float32
        )
        t = scale.double() * P[kb].double()
        acc64 = acc.double()
        s = t + acc64
        bb = s - t
        err = (t - (s - bb)) + (acc64 - bb)
        out = s.to(torch.float32)
        midpoint = (s.view(torch.int64) & ((1 << 29) - 1)) == (1 << 28)
        risky = (err != 0) & torch.isfinite(s) & (midpoint | (s.abs() < tiny))
        for m, n in risky.nonzero(as_tuple=False).tolist():
            out[m, n] = exact.fma(
                float(scale[m, n]), float(P[kb, m, n]), float(acc[m, n])
            )
        acc = out
    return acc


def reference(a, w, sa, sb, out_dtype=torch.bfloat16, P=None):
    acc = exact_promotion(block_partials(a, w) if P is None else P, sa, sb)
    return acc if out_dtype == torch.float32 else acc.to(out_dtype)


# ----------------------------------------------------------------------------------------------- device-free


def test_fingerprint_is_a_function_of_k_and_the_output_format():
    f = bi.fingerprint("gemm_fp8_nt_groupwise", K=7168, out_dtype=torch.bfloat16)
    assert f.startswith("gemm-e4m3-kblock128-") and f == bi.fingerprint(
        "gemm_fp8_nt_groupwise", K=7168, out_dtype="bf16"
    )
    # K and the output dtype are dimensions of the specification; N, M, the scale layout and the scales are not
    distinct = {
        bi.fingerprint("gemm_fp8_nt_groupwise", K=K, out_dtype=od)
        for K in (7168, 7168 + 128)
        for od in (torch.bfloat16, torch.float16)
    }
    assert len(distinct) == 4
    assert f != bi.fingerprint(
        "bmm_fp8", N=7168, K=7168, dtype=E4, out_dtype=torch.bfloat16
    ), "the block fold is a different class from the per-tensor fold"
    assert spec_fingerprint(7168, "bf16", 1, FMT) == f
    assert bi.class_name("gemm_fp8_nt_groupwise") == "gemm.e4m3.kblock128"
    with pytest.raises(TypeError):
        bi.fingerprint("gemm_fp8_nt_groupwise", K=7168)
    with pytest.raises(TypeError):
        bi.fingerprint("gemm_fp8_nt_groupwise", N=7168, K=7168, out_dtype="bf16")
    with pytest.raises(ValueError):
        bi.fingerprint("gemm_fp8_nt_groupwise", K=7168, out_dtype=torch.float32)
    with pytest.raises(ValueError):
        GemmSpec(256, "bf16", chunks=2, in_fmt=FMT)


def test_space_is_the_block_scale_tiles_on_a_k_major_weight():
    space = knob_product(in_fmt=FMT)
    assert len(space) == 336
    assert {k.tile_m for k in space} == {128, 256}
    assert {k.tile_n for k in space} == {16, 32, 64, 128, 256}
    assert all(not k.use_2cta_instrs for k in space if k.tile_n == 16)
    assert all(k.cluster_k == 1 and not k.swap_ab and not k.cta_chunks for k in space)
    assert len(enumerate_space(1000, 7168, 7168, b_major="k", in_fmt=FMT)) == 336
    assert enumerate_space(1000, 7168, 7168, b_major="n", in_fmt=FMT) == []
    assert enumerate_space(1000, 7168, 7168 + 64, b_major="k", in_fmt=FMT) == []
    for N, K in DS_PAIRS:
        assert chunks_for(N, K, FMT) == 1  # no split variant
    assert chunks_for(64, 65536, FMT) == 1


def test_blocked_fold_traces_the_indexed_block_digit():
    """The promotion issues the spec's ``block`` digit -- an indexed fold: the step takes the block's index for its scale -- one
    block per k-tile; the engine's loop-nest trace of it over the block partials is the digit's direct fold and the spec's
    canonical term, at one block per stage and at several."""
    from flashinfer.batch_invariant.canon import BlockedFold, Reduction
    from flashinfer.batch_invariant.fp_interface.term import TermAlgebra
    from flashinfer.gemm.batch_invariant.spec import reduction_blockscale

    alg = TermAlgebra()
    red = reduction_blockscale(4 * BLOCK_K, 0, 0, FMT)
    kstep, block = red.digits
    assert block.indexed and not kstep.indexed
    partials = [
        kstep.reduce(
            alg, [red.leaf(alg, {"block": kb, "kstep": t}) for t in range(kstep.radix)]
        )
        for kb in range(block.radix)
    ]
    direct = block.reduce(alg, partials)
    assert direct is red.canonical(alg)
    promotion = Reduction("promotion", (block,), lambda fp, idx: partials[idx["block"]])
    for inner in (1, 2):
        low = BlockedFold("block", inner)
        assert low.trace(promotion, alg, block.radix // inner) is direct, inner


@pytest.mark.parametrize("family,sm_count", PARTS)
def test_dispatcher_picks_are_members_of_the_space(family, sm_count):
    """R3 membership, and the structure the heuristic's block-scale terms give the pick at every nominal part: never the
    2-CTA pair (its promotion runs in lockstep, `heuristic.estimate_cycles`); at a few tokens the narrowest tile whose
    tiles fit the part in one wave (the per-block hand-off chain costs the same at every width) in the 4-wide N-cluster
    when its clusters fit (the token box's zero-fill shared by the multicast); at many rows a wide tile, 128 or 256 columns, once
    the 128-wide tiles alone fill the part."""
    cap4 = capacity_for(family, sm_count, nominal=True)[4]
    for N, K in DS_PAIRS:
        for M in DS_MS:
            space = set(enumerate_space(M, N, K, b_major="k", in_fmt=FMT))
            top = rank_topk(
                M, N, K, 1, "k", sm_count, 10, family, in_fmt=FMT, nominal=True
            )
            assert top and all(k in space for k in top), (family, sm_count, M, N, K)
            pick = default_config(M, N, K, 1, "k", sm_count, family, FMT, nominal=True)
            assert top[0] == pick
            where = (family, sm_count, M, N, K, pick.describe())
            assert not pick.use_2cta_instrs, where
            if M <= 32:
                # the narrowest tile whose tiles fit one wave, in 4-wide N-clusters when those fit the part's cluster capacity
                tiles = {tn: -(-N // tn) for tn in (16, 32, 64, 128, 256)}
                in_clusters = [tn for tn, t in tiles.items() if (t + 3) // 4 <= cap4]
                if in_clusters:
                    assert (pick.tile_n, pick.cluster_n) == (in_clusters[0], 4), where
                else:
                    assert (
                        pick.tile_n
                        == [tn for tn, t in tiles.items() if t <= sm_count][0]
                    ), where
            wide_tiles = -(-M // 128) * -(-N // 128)
            if M >= 1024 and wide_tiles >= 2 * sm_count:
                # the 128-wide tiles alone fill the part twice over: a wide tile, 128 or 256 columns (a narrower N, or a single
                # partial wave, keeps the SMs busy with narrower tiles)
                assert pick.tile_m in (128, 256) and pick.tile_n >= 128, where


def test_tactic_encoding_round_trips():
    for k in RECORDS:
        assert tuning.knobs_from_tactic(tuning.knobs_to_tactic(k)) == k


def test_flag_bypasses_the_operator_backend_selection():
    """Under ``batch_invariant=True`` the operator's auto-backend heuristic yields a non-empty list whatever the backends'
    availability: the flag path runs the class, so a missing backend never turns into a backend error."""
    from flashinfer.gemm.gemm_base import _heuristic_func_gemm_fp8_nt_groupwise

    a = torch.zeros(4, 256).to(E4)
    w = torch.zeros(128, 256).to(E4)
    sa, sb = torch.ones(4, 2), torch.ones(1, 2)
    assert _heuristic_func_gemm_fp8_nt_groupwise(
        [], a, w, sa, sb, "K", batch_invariant=True
    ) == ["batch_invariant"]


def test_unsupported_device_raises_before_any_kernel():
    a = torch.zeros(4, 256).to(E4)
    w = torch.zeros(128, 256).to(E4)
    sa, sb = torch.ones(4, 2), torch.ones(1, 2)
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        ours(a, w, sa, sb)


# ----------------------------------------------------------------------------------------------- the exact model on hardware


@needs_device
@pytest.mark.parametrize("M,N,K", EXACT_SHAPES)
def test_gpu_equals_the_exact_model(M, N, K):
    """The kernel's output is bit for bit the exact model's in every output format and both scale layouts (the dispatcher's
    record for the shape)."""
    a, w, sa, sb = operands(M, N, K, seed=11)
    P = block_partials(a, w)
    for od in (torch.float32, torch.bfloat16, torch.float16):
        ref = bits(reference(a, w, sa, sb, od, P=P))
        got_k = ours(a, w, sa, sb, out_dtype=od)
        got_mn = gemm_fp8_nt_groupwise_batch_invariant(
            a, w, sa.t().contiguous(), sb.t().contiguous(), "MN", out_dtype=od
        )
        assert torch.equal(bits(got_k).cpu(), ref), (M, N, K, od, "K layout")
        assert torch.equal(bits(got_mn).cpu(), ref), (M, N, K, od, "MN layout")


@needs_device
def test_exact_model_over_records_and_scale_corners():
    """Hand-picked records of the space (every tile N, both CTA groups, clusters, rasters, swizzles) on scales of a wide
    exponent range, exact powers of two and ones, bit for bit the exact model."""
    M, N, K = 300, 576, 1024
    a, w, _, _ = operands(M, N, K, seed=21)
    KB, NB = K // BLOCK_K, -(-N // SCALE_BLOCK_N)
    P = block_partials(a, w)
    corners = {
        "wide": (scales(M, KB, 22, span=20), scales(NB, KB, 23, span=20)),
        "pow2": (
            torch.exp2(torch.randint(-30, 30, (M, KB)).float()).cuda(),
            torch.exp2(torch.randint(-30, 30, (NB, KB)).float()).cuda(),
        ),
        "ones": (torch.ones(M, KB, device="cuda"), torch.ones(NB, KB, device="cuda")),
    }
    for name, (sa, sb) in corners.items():
        ref = bits(reference(a, w, sa, sb, torch.bfloat16, P=P))
        for k in RECORDS:
            got = ours(a, w, sa, sb, config=k)
            assert torch.equal(bits(got).cpu(), ref), (name, k.describe())


@needs_device
def test_block_fold_is_a_different_class_from_the_per_tensor_fold():
    """With unit scales the class is ``fold_kb fma(1, P_kb, acc)``: the block partials summed in block order -- not the
    per-tensor class's single fold over all K steps (a different association), as the fingerprints say; at K = 128 (one
    block) the two agree up to the sign of an exact zero (``fma(1, -0, +0) = +0``)."""
    M, N, K = 64, 1024, 2048
    a, w, _, _ = operands(M, N, K, seed=9)
    one_a = torch.ones(M, K // BLOCK_K, device="cuda")
    one_b = torch.ones(N // SCALE_BLOCK_N, K // BLOCK_K, device="cuda")
    unit = torch.ones(1, device="cuda")
    blk = ours(a, w, one_a, one_b, out_dtype=torch.float32)
    fold = mm_fp8_batch_invariant(a, w.t(), unit, unit, out_dtype=torch.float32)
    assert not torch.equal(bits(blk), bits(fold))
    a1, w1 = a[:, :BLOCK_K].contiguous(), w[:, :BLOCK_K].contiguous()
    blk1 = ours(
        a1,
        w1,
        one_a[:, :1].contiguous(),
        one_b[:, :1].contiguous(),
        out_dtype=torch.float32,
    )
    fold1 = mm_fp8_batch_invariant(a1, w1.t(), unit, unit, out_dtype=torch.float32)
    diff = bits(blk1) != bits(fold1)
    assert torch.all((fold1 == 0) | ~diff)
    assert not diff.any() or torch.all(blk1[diff] == 0)


@needs_device
def test_public_api_runs_the_class_and_both_layouts_agree():
    """``flashinfer.gemm_fp8_nt_groupwise(..., backend="auto", batch_invariant=True)`` is the class (the direct entry's bits),
    in the K and the MN scale layouts alike (R4), for both output dtypes."""
    for M, N, K in [(300, 576, 1024), (16, 7168, 7168)]:
        a, w, sa, sb = operands(M, N, K, seed=31)
        for od in (torch.bfloat16, torch.float16):
            ref = bits(ours(a, w, sa, sb, out_dtype=od))
            assert torch.equal(bits(api(a, w, sa, sb, "K", out_dtype=od)), ref)
            assert torch.equal(bits(api(a, w, sa, sb, "MN", out_dtype=od)), ref)
            out = torch.empty(M, N, dtype=od, device="cuda")
            assert api(a, w, sa, sb, "K", out=out) is out and torch.equal(
                bits(out), ref
            )


@needs_device
def test_k_native_scale_staging_is_a_placement():
    """The ``K`` layout's two stagings -- the K-native ring (four blocks per stage, a row's four scales one 16-byte copy from
    the caller's tensor) and the per-block ring over the MN-major copy -- and the ``MN`` layout compute the same bits at
    every shape of the DeepSeek grid, forced either way; the API's own choice (``native_k_staging``: a few rows) and a
    K-layout tensor whose base is not 16-byte aligned (handed over as its copy) likewise. The staging changes which bytes
    a ring stage holds, never the fold."""
    for N, K in DS_PAIRS:
        for M in DS_MS:
            a, w, sa, sb = operands(M, N, K, seed=81)
            ref = bits(
                gemm_fp8_nt_groupwise_batch_invariant(
                    a, w, sa.t().contiguous(), sb.t().contiguous(), "MN"
                )
            )
            assert torch.equal(bits(ours(a, w, sa, sb, native_k=True)), ref), (
                "native",
                M,
                N,
                K,
            )
            assert torch.equal(bits(ours(a, w, sa, sb, native_k=False)), ref), (
                "copy",
                M,
                N,
                K,
            )
            assert torch.equal(bits(ours(a, w, sa, sb)), ref), ("default", M, N, K)
            del a, w, sa, sb, ref
    assert native_k_staging(64) and not native_k_staging(65)
    M, N, K = 64, 7168, 7168
    a, w, sa, sb = operands(M, N, K, seed=83)
    # a 4-byte-aligned base (a column view): not the 16-byte copies'; the API hands the copy over
    sa_off = torch.empty(M, K // BLOCK_K + 1, device="cuda")[:, 1:]
    sa_off.copy_(sa)
    assert sa_off.data_ptr() % 16 != 0
    ref = bits(ours(a, w, sa, sb, native_k=True))
    assert torch.equal(bits(ours(a, w, sa_off, sb)), ref)
    assert torch.equal(bits(api(a, w, sa, sb, "K")), ref)
    ref32 = bits(reference(a, w, sa, sb, torch.float32))
    assert torch.equal(
        bits(ours(a, w, sa, sb, native_k=True, out_dtype=torch.float32)).cpu(), ref32
    )
    # the operator (its autotuner path) takes the staging the rule names: the K layout as it is at a few rows, the MN
    # copy at more, the MN layout never native
    from flashinfer.gemm.batch_invariant.blocks import compile_fp8bs

    seen = []
    real = compile_fp8bs.get_compiled_fp8bs

    def recording(cfg, clusters, out_fmt="bf16", family="", native_k=False):
        seen.append(native_k)
        return real(cfg, clusters, out_fmt, family, native_k)

    compile_fp8bs.get_compiled_fp8bs = recording
    try:
        for M, mode, want in ((16, "K", True), (128, "K", False), (16, "MN", False)):
            a, w, sa, sb = operands(M, 2048, 1024, seed=85)
            seen.clear()
            api(a, w, sa, sb, mode)
            assert seen and all(v == want for v in seen), (M, mode, seen)
    finally:
        compile_fp8bs.get_compiled_fp8bs = real


@needs_device
@pytest.mark.parametrize("M,N,K", NATIVE_TAIL_SHAPES)
def test_k_native_staging_short_last_stage_equals_the_exact_model(M, N, K):
    """At a few rows the ``K`` layout's scales are staged natively, four blocks per ring stage and the short last group block by
    block: a K of one, two or three blocks is that short group alone, six or seven a whole stage and then it. The kernel's
    output is bit for bit the exact model's in every output format, the staging forced and the API's own choice."""
    from flashinfer.gemm.batch_invariant.blocks.blockscale_fp8 import (
        SCALE_GROUP_NATIVE_K,
    )

    assert native_k_staging(M) and (K // BLOCK_K) % SCALE_GROUP_NATIVE_K
    a, w, sa, sb = operands(M, N, K, seed=13)
    P = block_partials(a, w)
    for od in (torch.float32, torch.bfloat16, torch.float16):
        ref = bits(reference(a, w, sa, sb, od, P=P))
        got_forced = ours(a, w, sa, sb, native_k=True, out_dtype=od)
        got_default = ours(a, w, sa, sb, out_dtype=od)
        assert torch.equal(bits(got_forced).cpu(), ref), (M, N, K, od, "native")
        assert torch.equal(bits(got_default).cpu(), ref), (M, N, K, od, "default")


@needs_device
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs a second GPU")
def test_entry_follows_the_operands_device():
    """The compile and the launch run on A's device whatever the current one."""
    a, w, sa, sb = operands(37, 2048, 1024, seed=15)
    ref = bits(ours(a, w, sa, sb))
    with torch.cuda.device(1):
        assert torch.equal(bits(ours(a, w, sa, sb)), ref)


# ----------------------------------------------------------------------------------------------- R1 / R2 / R3


@needs_device
@pytest.mark.parametrize("M,N,K", [(1000, 4096, 2048), (37, 2048, 1024)])
def test_determinism_and_graph_replay(M, N, K):
    a, w, sa, sb = operands(M, N, K, seed=51)
    ref = bits(ours(a, w, sa, sb)).clone()
    for _ in range(3):
        assert torch.equal(bits(ours(a, w, sa, sb)), ref)
        assert torch.equal(bits(api(a, w, sa, sb, "MN")), ref)
    s = torch.cuda.Stream()
    buf = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    with torch.cuda.stream(s):
        ours(a, w, sa, sb, out=buf)  # warm-up on the capture stream
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            ours(a, w, sa, sb, out=buf)
    torch.cuda.synchronize()
    for _ in range(2):
        buf.zero_()
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(bits(buf), ref)


@needs_device
def test_batch_permutation_subbatch_invariance():
    """R2: a row's bits do not depend on the row count, its position or its neighbours; its block scales travel with it."""
    N, K = 2048, 4096
    a, w, sa, sb = operands(512, N, K, seed=41)
    full = bits(ours(a, w, sa, sb))
    for M in (1, 3, 16, 37, 64, 65, 128, 256, 500):
        assert torch.equal(
            bits(ours(a[:M].contiguous(), w, sa[:M].contiguous(), sb)), full[:M]
        ), f"M={M}"
    perm = torch.randperm(512, generator=torch.Generator().manual_seed(1)).cuda()
    assert torch.equal(
        bits(ours(a[perm].contiguous(), w, sa[perm].contiguous(), sb)), full[perm]
    )
    for k in (1, 2, 8, 37):
        idx = torch.randperm(512, generator=torch.Generator().manual_seed(k))[:k].cuda()
        assert torch.equal(
            bits(ours(a[idx].contiguous(), w, sa[idx].contiguous(), sb)), full[idx]
        ), f"k={k}"
    x = torch.cat([adv_fp8(5, K, seed=43), a[100:101], adv_fp8(7, K, seed=44)])
    sx = torch.cat(
        [scales(5, K // BLOCK_K, 45), sa[100:101], scales(7, K // BLOCK_K, 46)]
    )
    assert torch.equal(bits(ours(x, w, sx, sb))[5], full[100])


@needs_device
@pytest.mark.parametrize("M,N,K", [(1000, 4096, 2048), (37, 2048, 1024)])
def test_config_agreement(M, N, K):
    """R3 (the hardware hypothesis): the dispatcher's candidates for every part and the hand-picked records compute the
    exact model's bits."""
    a, w, sa, sb = operands(M, N, K, seed=61)
    ref = bits(reference(a, w, sa, sb))
    seen = set()
    for family, sm_count in PARTS:
        seen.update(
            rank_topk(M, N, K, 1, "k", sm_count, 10, family, in_fmt=FMT, nominal=True)
        )
    seen.update(k for k in RECORDS if can_implement(k, M, N, K, "k", 1, FMT))
    for k in sorted(seen, key=lambda k: k.describe()):
        assert torch.equal(bits(ours(a, w, sa, sb, config=k)).cpu(), ref), k.describe()
    assert len(seen) >= 16


@needs_device
@pytest.mark.parametrize("M,N,K", FULL_SPACE_SHAPES)
def test_whole_space_agreement(M, N, K, request):
    """Every record of the enumerated space (336) computes the exact model's bits; the whole space under ``--full``, a
    sample otherwise (the instruction variants are covered by `test_config_agreement` every run)."""
    space = enumerate_space(M, N, K, b_major="k", in_fmt=FMT)
    assert len(space) == 336
    full = False
    # registered by tests/conftest.py's parametrize plugin; absent when the file runs outside the repo's conftest
    with contextlib.suppress(ValueError):
        full = bool(request.config.getoption("--full"))
    records = space if full else space[:: len(space) // 12]
    a, w, sa, sb = operands(M, N, K, seed=71)
    ref = bits(reference(a, w, sa, sb))
    for k in records:
        assert torch.equal(bits(ours(a, w, sa, sb, config=k)).cpu(), ref), k.describe()


@needs_device
def test_persistent_grid_size_does_not_change_bits():
    """R2 for the SM count: a smaller persistent grid is what a smaller part, an SM carveout, MPS or a green context
    produce; it only decides which CTA computes which tile."""
    for M, N, K in [(1000, 4096, 2048), (64, 576, 7168)]:
        a, w, sa, sb = operands(M, N, K, seed=91)
        base = default_config(M, N, K, 1, "k", 204, "sm_107", FMT)
        ref = bits(ours(a, w, sa, sb, config=base))
        for g in (1, 3, 17, 74):
            cfg = dataclasses.replace(base, grid_clusters=g)
            assert torch.equal(bits(ours(a, w, sa, sb, config=cfg)), ref), f"grid={g}"


@needs_device
@pytest.mark.parametrize("N,K,mode", [(4096, 2048, "K"), (576, 7168, "MN")])
def test_autotuning_profiles_every_candidate_and_keeps_the_bits(N, K, mode):
    """Under ``autotune(True)`` the FlashInfer autotuner profiles the class's candidates (none may be dropped as
    unsupported) and the tuned process computes the same bits as the untuned one, during and after tuning."""
    a, w, sa, sb = operands(300, N, K, seed=101)

    def run(M):
        return bits(api(a[:M].contiguous(), w, sa[:M].contiguous(), sb, mode))

    ref = [run(M) for M in (1, 16, 300)]
    tuner = AutoTuner.get()
    tuner.clear_cache()  # no earlier call answers the profiles from the cache: the candidates are profiled in this test

    def failed():
        return {
            k: len(v)
            for k, v in tuner.stats.failed_tactics.items()
            if tuning.CUSTOM_OP_FP8BS in k
        }

    before = failed()
    profiled = tuner.stats.tuned_op_successful_configs.get(tuning.CUSTOM_OP_FP8BS, 0)
    with autotune(True):
        tuned = [run(M) for M in (1, 16, 300)]
    assert failed() == before, f"tactics failed while profiling: {failed()}"
    assert (
        tuner.stats.tuned_op_successful_configs.get(tuning.CUSTOM_OP_FP8BS, 0)
        > profiled
    )  # profiled here, not a cache hit
    post = [run(M) for M in (1, 16, 300)]
    for r, t, p in zip(ref, tuned, post, strict=True):
        assert torch.equal(r, t) and torch.equal(r, p)
    assert torch.equal(post[1], post[2][:16])


@needs_device
@pytest.mark.parametrize(
    "M,N,K", [(1000, 4096, 2048), (37, 2048, 1024), (16, 7168, 7168)]
)
def test_flashinfer_backends_within_tolerance(M, N, K):
    """FlashInfer's ``gemm_fp8_nt_groupwise`` backends (CUTLASS in both scale layouts; the cute-dsl SM107 kernel where it
    runs the shape) compute the same mathematics with their own association: an observation within a tolerance, not a
    member of the class. Their fp32 sum differs from the exact model's by at most a few ulps of the largest block term."""
    a, w, sa, sb = operands(M, N, K, seed=111)
    ref = exact_promotion(block_partials(a, w), sa, sb)
    scale = (
        (a.float().abs() @ w.float().abs().t()).cpu()
        * float(sa.max())
        * float(sb.max())
    )
    op = flashinfer.gemm.gemm_fp8_nt_groupwise
    major, minor = torch.cuda.get_device_capability(a.device)
    # the combinations a backend takes, settled before the call: CUTLASS in the K layout at any M and in the MN layout at M a
    # multiple of 4 (the operator's documented row requirement, which its K-layout kernel does not enforce); the cute-dsl kernel
    # on SM107, the MN layout, M a multiple of 4. Every error of a call is the test's failure
    aligned = M % 4 == 0
    combos = [("cutlass", "K")] + ([("cutlass", "MN")] if aligned else [])
    if op.is_backend_supported("cute-dsl", major * 10 + minor) and aligned:
        combos.append(("cute-dsl", "MN"))
    ran = 0
    for be, mode in combos:
        if mode == "K":
            c = op(
                a, w, sa, sb, scale_major_mode="K", out_dtype=torch.bfloat16, backend=be
            )
        else:
            c = op(
                a,
                w,
                sa.t().contiguous(),
                sb.t().contiguous(),
                scale_major_mode="MN",
                out_dtype=torch.bfloat16,
                backend=be,
            )
        err = ((c.float().cpu() - ref).abs() / (scale + 1e-30)).max().item()
        assert err <= 2.0**-7, (
            be,
            mode,
            err,
        )  # a bf16 output's own rounding bounds the norm-relative difference
        ran += 1
    assert ran == len(combos) >= 1


@needs_device
def test_trace_apply_does_not_replace_the_class():
    """A Trace Apply solution registered for the call's definition name replaces the plain API call (the control); under
    ``batch_invariant=True`` it must not: the flag fixes the call's dataflow to the class."""
    import flashinfer.trace_apply as ta
    from flashinfer.trace_apply.apply import (
        _registry_by_fi_api,
        bind_namespace,
        build_extractor_maps,
        extract_axes,
    )

    M, N, K = 16, 4096, 2048
    a, w, sa, sb = operands(M, N, K, seed=171)
    ref = bits(api(a, w, sa, sb)).clone()
    reg = _registry_by_fi_api()
    key = next(k for k in reg if k.endswith(".gemm_fp8_nt_groupwise"))
    original, templates = reg[key]
    ns = bind_namespace(original, (a, w, sa, sb), {"scale_major_mode": "K"})
    name = templates[0].definition_name(
        extract_axes(build_extractor_maps(templates), ns)
    )
    calls: list = []

    def solution(*args, **kwargs):
        calls.append(1)
        return torch.full((M, N), 2.0, dtype=torch.bfloat16, device="cuda")

    ta.enable_apply({name: solution})
    try:
        plain = flashinfer.gemm.gemm_fp8_nt_groupwise(
            a, w, sa, sb, scale_major_mode="K"
        )
        assert calls and bool(torch.all(plain == 2.0)), (
            "the control: the solution replaces the plain call"
        )
        seen = len(calls)
        got = api(a, w, sa, sb)
        assert len(calls) == seen, (
            "the registered solution ran under batch_invariant=True"
        )
        assert torch.equal(bits(got), ref)
        assert ta.stats().get(key, {}).get("fallback_batch_invariant", 0) >= 1
    finally:
        ta.disable_apply()


@needs_device
def test_unsupported_inputs_raise_and_nothing_falls_back():
    M, N, K = 16, 256, 512
    a, w, sa, sb = operands(M, N, K, seed=81)
    E = bi.BatchInvariantUnsupportedError
    with pytest.raises(E, match="float8_e4m3fn"):
        ours(a.to(torch.float16).to(E5), w.to(torch.float16).to(E5), sa, sb)
    with pytest.raises(E, match="multiple of 128"):
        ours(
            adv_fp8(M, 576, seed=1),
            adv_fp8(N, 576, seed=2, cancel=False),
            scales(M, 5, 3),
            scales(2, 5, 4),
        )
    with pytest.raises(E, match="row-major"):
        ours(a, w.t().contiguous().t(), sa, sb)  # an N-major weight
    with pytest.raises(E, match="a_scale"):
        ours(a, w, sa.t().contiguous(), sb)
    with pytest.raises(E, match="b_scale"):
        ours(a, w, sa, sb[:1].contiguous())
    with pytest.raises(E, match="float32"):
        ours(a, w, sa.to(torch.bfloat16), sb)
    with pytest.raises(E, match="device"):
        ours(a, w, sa.cpu(), sb)
    with pytest.raises(E, match="scale_major_mode"):
        gemm_fp8_nt_groupwise_batch_invariant(a, w, sa, sb, "NK")
    with pytest.raises(E, match="multiple of 8"):
        ours(a, adv_fp8(260, K, seed=5, cancel=False), sa, scales(3, 4, 6))
    with pytest.raises(E, match="enumerated space"):
        ours(a, w, sa, sb, config=Knobs((128, 192), (1, 1), False, True, "m", 1))
    with pytest.raises(E, match="scale_granularity_mnk"):
        flashinfer.gemm.gemm_fp8_nt_groupwise(
            a,
            w,
            sa,
            sb,
            scale_major_mode="K",
            scale_granularity_mnk=(1, 64, 128),
            backend="auto",
            batch_invariant=True,
        )
    with pytest.raises(E, match='backend="auto"'):
        flashinfer.gemm.gemm_fp8_nt_groupwise(
            a, w, sa, sb, scale_major_mode="K", backend="cutlass", batch_invariant=True
        )
    with pytest.raises(ValueError):  # the operator's own output-dtype check runs first
        flashinfer.gemm.gemm_fp8_nt_groupwise(
            a,
            w,
            sa,
            sb,
            scale_major_mode="K",
            out_dtype=torch.float32,
            backend="auto",
            batch_invariant=True,
        )
    for bad_a, bad_w in ((a[None], w), (a, adv_fp8(N, 2 * K, seed=5, cancel=False))):
        with pytest.raises(
            ValueError
        ):  # likewise its shape check: a non-2-D operand, a K mismatch
            flashinfer.gemm.gemm_fp8_nt_groupwise(
                bad_a,
                bad_w,
                sa,
                sb,
                scale_major_mode="K",
                backend="auto",
                batch_invariant=True,
            )
    out = ours(a[:0], w, sa[:0], sb)
    assert out.shape == (0, N) and out.dtype == torch.bfloat16
    out = torch.empty(M, N, dtype=torch.float16, device="cuda")
    assert ours(a, w, sa, sb, out=out, out_dtype=torch.float16) is out
    with pytest.raises(E, match="out"):
        ours(
            a,
            w,
            sa,
            sb,
            out=torch.empty(M, N, dtype=torch.bfloat16, device="cuda")[:, ::1]
            .t()
            .contiguous()
            .t(),
        )
