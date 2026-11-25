# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Batch-invariant groupwise FP8 GEMM, including an exact reference for scale promotion and both scale layouts."""

import pytest
import torch

import flashinfer
from flashinfer import batch_invariant as bi
from flashinfer.autotuner import autotune
from flashinfer.batch_invariant.fp_interface import exact
from flashinfer.gemm.batch_invariant.api import (
    gemm_fp8_nt_groupwise_batch_invariant,
)
from flashinfer.gemm.batch_invariant.generator import (
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
needs_device = pytest.mark.usefixtures("cudnn_batch_invariant_device")

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


def operands(M: int, N: int, K: int, seed: int = 0, *, exact_blocks=False):
    """``a (M, K)``, the row-major weight ``w (N, K)``, ``sa (M, KB)`` and ``sb (ceil(N / 128), KB)`` in the K layout."""
    a = adv_fp8(M, K, seed=seed + 1)
    w = adv_fp8(N, K, seed=seed + 2, cancel=False)
    if exact_blocks:
        # Each 128-wide dot is then an exactly representable integer on both the MMA and the reference.
        a = a.float().round().clamp(-4, 4).to(E4)
        w = w.float().round().clamp(-4, 4).to(E4)
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


def block_partials(a, w):
    """Exact, unscaled dot products of the integer-valued E4M3 blocks used by the promotion oracle."""
    m, k = a.shape
    n = w.shape[0]
    lhs = a.double().reshape(m, k // BLOCK_K, BLOCK_K).transpose(0, 1)
    rhs = w.double().reshape(n, k // BLOCK_K, BLOCK_K).permute(1, 2, 0)
    return torch.bmm(lhs, rhs).float()


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
    a, w, sa, sb = operands(M, N, K, seed=11, exact_blocks=True)
    P = block_partials(a, w)
    for od in (torch.bfloat16, torch.float16):
        ref = bits(reference(a, w, sa, sb, od, P=P))
        got_k = ours(a, w, sa, sb, out_dtype=od)
        got_mn = gemm_fp8_nt_groupwise_batch_invariant(
            a, w, sa.t().contiguous(), sb.t().contiguous(), "MN", out_dtype=od
        )
        assert torch.equal(bits(got_k).cpu(), ref), (M, N, K, od, "K layout")
        assert torch.equal(bits(got_mn).cpu(), ref), (M, N, K, od, "MN layout")


@needs_device
def test_exact_model_over_scale_corners():
    """Check wide, power-of-two and unit scales against an independently rounded promotion."""
    M, N, K = 300, 576, 1024
    a, w, _, _ = operands(M, N, K, seed=21, exact_blocks=True)
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
        for mode in ("K", "MN"):
            got = api(a, w, sa, sb, mode)
            assert torch.equal(bits(got).cpu(), ref), (name, mode)


@needs_device
def test_public_api_runs_the_class_and_both_layouts_agree():
    """``flashinfer.gemm_fp8_nt_groupwise(..., backend="auto", batch_invariant=True)`` is checked against exact scale promotion,
    in the K and the MN scale layouts alike (R4), for both output dtypes."""
    for M, N, K in [(300, 576, 1024), (16, 7168, 7168)]:
        a, w, sa, sb = operands(M, N, K, seed=31, exact_blocks=True)
        for od in (torch.bfloat16, torch.float16):
            ref = bits(reference(a, w, sa, sb, od)).to(a.device)
            assert torch.equal(bits(api(a, w, sa, sb, "K", out_dtype=od)), ref)
            assert torch.equal(bits(api(a, w, sa, sb, "MN", out_dtype=od)), ref)
            out = torch.empty(M, N, dtype=od, device="cuda")
            assert api(a, w, sa, sb, "K", out=out) is out and torch.equal(
                bits(out), ref
            )


@needs_device
@pytest.mark.parametrize("M,N,K", NATIVE_TAIL_SHAPES)
def test_k_native_staging_short_last_stage_equals_the_exact_model(M, N, K):
    """One to three scale blocks in the final stage must give the same bits as exact promotion."""
    a, w, sa, sb = operands(M, N, K, seed=13, exact_blocks=True)
    P = block_partials(a, w)
    for od in (torch.bfloat16, torch.float16):
        ref = bits(reference(a, w, sa, sb, od, P=P))
        got_default = ours(a, w, sa, sb, out_dtype=od)
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
@pytest.mark.parametrize("N,K,mode", [(4096, 2048, "K"), (576, 7168, "MN")])
def test_public_rows_keep_their_bits_inside_autotune(N, K, mode):
    """The public batch-invariant entry agrees before, during and after autotuning."""
    a, w, sa, sb = operands(300, N, K, seed=101)

    def run(M):
        return bits(api(a[:M].contiguous(), w, sa[:M].contiguous(), sb, mode))

    ref = [run(M) for M in (1, 16, 300)]
    with autotune(True):
        tuned = [run(M) for M in (1, 16, 300)]
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
@pytest.mark.parametrize("M,N,K", [(16, 4096, 4096), (300, 576, 1024)])
def test_public_output_buffer_and_graph_replay(M, N, K):
    """A caller-provided output survives graph replay and must be disjoint from the inputs."""
    a, w, sa, sb = operands(M, N, K, seed=181, exact_blocks=True)
    ref = bits(reference(a, w, sa, sb)).to(a.device)
    for mode in ("K", "MN"):
        assert torch.equal(bits(api(a, w, sa, sb, mode)), ref)
        out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
        assert api(a, w, sa, sb, mode, out=out) is out and torch.equal(bits(out), ref)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        api(a, w, sa, sb, "K", out=out)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            api(a, w, sa, sb, "K", out=out)
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
        api(alias_a, w, sa, sb, "K", out=alias_out)


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
    with pytest.raises(E, match="cuDNN"):
        ours(a, w, sa, sb, native_k=True)
    with pytest.raises(E, match="scale_major_mode"):
        gemm_fp8_nt_groupwise_batch_invariant(a, w, sa, sb, "NK")
    with pytest.raises(E, match="multiple of 8"):
        ours(a, adv_fp8(260, K, seed=5, cancel=False), sa, scales(3, 4, 6))
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
