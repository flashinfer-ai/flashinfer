# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The batch-invariant fp8 GEMM: ``flashinfer.bmm_fp8(A, B, A_scale, B_scale, dtype, backend="auto", batch_invariant=True)``.

The classes ``gemm.e4m3`` / ``gemm.e5m2`` (``kfold.scaled``, ``kchunk[C].scaled``) are the bf16 class's dataflow at the
32-wide ``kind::f8f6f4`` step with the chain ``cvt(mul.rn(acc, mul.rn(a_scale, b_scale)))``. There is no exact CPU oracle
for the MMA step, so the classes are checked by bit agreement across their configurations and the two implementations of
the split class (the hardware hypothesis of ``lean/BIKernels/GEMMConfig.lean``), by the launch-model properties R1-R2 on
adversarial operands, along the rows and along the batch index, by the exact chain (the scaled output against the fp32
accumulator scaled on the CPU with the same correctly rounded operations), and by an accuracy yardstick against the
fp8 backends of ``bmm_fp8``. Device-free tests cover the fingerprints, the chunk rule, the dispatcher and the tuner's
tactic encoding; the GPU tests need SM100 / SM103 / SM107.
"""

import dataclasses

import pytest
import torch

import flashinfer
from flashinfer import batch_invariant as bi
from flashinfer.autotuner import AutoTuner, autotune
from flashinfer.gemm.batch_invariant import tuning
from flashinfer.gemm.batch_invariant.api import (
    BMM_FP8_ARCHS,
    bmm_fp8_batch_invariant,
    mm_fp8_batch_invariant,
)
from flashinfer.gemm.batch_invariant.blocks.operands import can_implement
from flashinfer.gemm.batch_invariant.heuristic import PARTS
from flashinfer.gemm.batch_invariant.generator import (
    REFERENCE_SHAPE,
    Knobs,
    enumerate_space,
    knob_product,
)
from flashinfer.gemm.batch_invariant.schedule import (
    chunks_for,
    default_config,
    rank_topk,
)

E4, E5 = torch.float8_e4m3fn, torch.float8_e5m2
_supported = torch.cuda.is_available() and bi.is_supported_device(
    torch.device("cuda"), BMM_FP8_ARCHS
)
needs_device = pytest.mark.skipif(
    not _supported,
    reason="needs a CUDA device of SM100 / SM103 / SM107 with the CuTe DSL",
)


def bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16)


def adv_fp8(*shape, seed: int, dt=E4, cancel: bool = True) -> torch.Tensor:
    """Adversarial fp8 operand: the format's exponent range, random signs, near-cancelling neighbour pairs, zeros of
    both signs, so that any accumulation-order or scaling difference reaches the output bits."""
    g = torch.Generator().manual_seed(seed)
    span = 7 if dt == E4 else 14
    e = (torch.rand(*shape, generator=g) * 2 - 1) * span
    x = (torch.rand(*shape, generator=g) * 2 - 1) * torch.exp2(e)
    if cancel and shape[-1] >= 8:
        k = torch.arange(0, shape[-1] - 1, 8)
        x[..., k + 1] = -x[..., k] * (1 + 2.0**-3)
    z = torch.randint(0, 61, shape, generator=g)
    x[z == 0] = 0.0
    x[z == 1] = -0.0
    return x.to(dt).cuda()


def weights(b: int, N: int, K: int, seed: int, dt=E4) -> torch.Tensor:
    """``bmm_fp8``'s weight: (b, K, N) column-major, the transpose of a row-major (b, N, K) tensor (the kernel's ``k`` layout)."""
    return adv_fp8(b, N, K, seed=seed, dt=dt, cancel=False).transpose(1, 2)


def scales(sa: float = 2**-3 * 1.3, sb: float = 2**2 * 0.7):
    return (
        torch.tensor([sa], dtype=torch.float32, device="cuda"),
        torch.tensor([sb], dtype=torch.float32, device="cuda"),
    )


def bi_bmm(A, B, sa, sb, dtype=torch.bfloat16, **kw):
    return flashinfer.bmm_fp8(
        A, B, sa, sb, dtype, backend="auto", batch_invariant=True, **kw
    )


def ref64(a, b, sa, sb):
    return (a.float().double() @ b.float().double()) * (sa.double() * sb.double())


def norm_rel_err(c, a, b, sa, sb) -> float:
    ref = ref64(a, b, sa, sb)
    scale = (a.float().double().abs() @ b.float().double().abs()) * abs(
        float(sa) * float(sb)
    )
    return ((c.double() - ref).abs() / (scale + 1e-30)).max().item()


# ----------------------------------------------------------------------------------------------- device-free


def test_fingerprint_is_a_function_of_the_weight_shape_and_formats():
    f = bi.fingerprint("bmm_fp8", N=4096, K=4096, dtype=E4, out_dtype=torch.bfloat16)
    assert f.startswith("gemm-e4m3-kfold-scaled-") and f == bi.fingerprint(
        "bmm_fp8", N=4096, K=4096, dtype="e4m3", out_dtype="bf16"
    )
    assert bi.fingerprint(
        "bmm_fp8", N=128, K=4096, dtype=E4, out_dtype=torch.bfloat16
    ).startswith("gemm-e4m3-kchunk2-scaled-")
    assert bi.fingerprint(
        "bmm_fp8", N=64, K=8192, dtype=E5, out_dtype=torch.float16
    ).startswith("gemm-e5m2-kchunk8-scaled-")
    # the input dtype, the output dtype and the raw K are dimensions of the specification; the batch and M are not
    distinct = {
        bi.fingerprint("bmm_fp8", N=4096, K=K, dtype=dt, out_dtype=od)
        for K in (4096, 4096 + 32)
        for dt in (E4, E5)
        for od in (torch.bfloat16, torch.float16)
    }
    assert len(distinct) == 8
    assert f != bi.fingerprint("mm_bf16", N=4096, K=4096)
    assert bi.class_name("bmm_fp8") == "gemm.fp8"
    with pytest.raises(TypeError):
        bi.fingerprint("bmm_fp8", N=4096, K=4096, dtype=E4)
    with pytest.raises(TypeError):
        bi.fingerprint(
            "bmm_fp8", N=4096, K=4096, dtype=E4, out_dtype=torch.bfloat16, b=4
        )
    with pytest.raises(ValueError):
        bi.fingerprint(
            "bmm_fp8", N=4096, K=4096, dtype=torch.bfloat16, out_dtype=torch.bfloat16
        )


def test_chunk_rule_takes_no_m_and_counts_32_wide_steps():
    import inspect

    params = list(inspect.signature(chunks_for).parameters)
    assert params == ["N", "K", "in_fmt"] and "M" not in params
    assert chunks_for(4096, 4096, "e4m3") == 1 and chunks_for(8192, 8192, "e5m2") == 1
    assert chunks_for(128, 4096, "e4m3") == 2 and chunks_for(64, 8192, "e4m3") == 8
    # a chunk is a whole number of 32-wide steps: where the bf16 rule splits 16640 into 16 chunks of 1040 (65 bf16 steps),
    # the fp8 rule stops at 8 (1040 is not a multiple of 32)
    assert chunks_for(64, 16640) == 16 and chunks_for(64, 16640, "e4m3") == 8
    for N, K in [(64, 64), (512, 65536), (2048, 2048), (64, 8208), (8, 1040)]:
        C = chunks_for(N, K, "e4m3")
        assert C & (C - 1) == 0 and (C == 1 or (K % (32 * C) == 0 and K // C >= 256))


def test_fp8_space_has_no_8_wide_tile():
    """``kind::f8f6f4`` admits tile N >= 16: 840 fp8 records against the bf16 class's 888; an MN-major B drops the 2-CTA
    16-wide decode tile (816), a K-major one keeps it."""
    assert len(knob_product(in_fmt="e4m3")) == 840 and len(knob_product()) == 888
    assert all(k.tile_n >= 16 for k in knob_product(in_fmt="e5m2"))
    assert len(enumerate_space(*REFERENCE_SHAPE, b_major="k", in_fmt="e4m3")) == 840
    assert len(enumerate_space(*REFERENCE_SHAPE, b_major="n", in_fmt="e4m3")) == 816


@pytest.mark.parametrize("family,sm_count", PARTS)
def test_dispatcher_picks_are_members_of_the_space(family, sm_count):
    """R3 membership: every configuration the dispatcher or the tuner's candidate list returns for the fp8 classes is a
    well-formed record the kernel accepts for the shape, format and layout; the candidates are distinct and the default
    is their first."""
    spaces = {
        lay: set(enumerate_space(*REFERENCE_SHAPE, b_major=lay, in_fmt="e4m3"))
        for lay in ("n", "k")
    }
    for M in (1, 16, 300, 4096):
        for N, K in [
            (4096, 4096),
            (14336, 4096),
            (128, 4096),
            (64, 8192),
            (2048, 16384),
        ]:
            C = chunks_for(N, K, "e4m3")
            for b_major in ("n", "k"):
                top = rank_topk(M, N, K, C, b_major, sm_count, 10, family, "e4m3")
                assert 1 <= len(top) <= 10 and len(set(top)) == len(top)
                assert top[0] == default_config(
                    M, N, K, C, b_major, sm_count, family, "e4m3"
                )
                for cfg in top:
                    assert cfg.well_formed("e4m3") and can_implement(
                        cfg, M, N, K, b_major, C, "e4m3"
                    ), cfg.describe()
                    if not cfg.cta_chunks:
                        assert cfg in spaces[b_major], cfg.describe()


def test_tactic_encoding_round_trips():
    for cfg in enumerate_space(16, 128, 8192, chunks=4, b_major="k", in_fmt="e4m3")[
        ::97
    ]:
        t = tuning.knobs_to_tactic(cfg)
        assert (
            tuning.knobs_from_tactic(t) == cfg
            and tuning.knobs_from_tactic(list(t)) == cfg
        )


def test_unsupported_device_raises_before_any_kernel():
    a = torch.zeros(2, 4, 64).to(E4)
    w = torch.zeros(2, 64, 64).to(E4).transpose(1, 2)
    s = torch.ones(1)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="gemm.fp8"):
        bmm_fp8_batch_invariant(a, w, s, s, torch.bfloat16)


# ----------------------------------------------------------------------------------------------- on the device


@needs_device
@pytest.mark.parametrize(
    "M,N,K,dt", [(1000, 4096, 4096, E4), (512, 4000, 1024, E5), (300, 128, 4096, E4)]
)
def test_matches_reference(M, N, K, dt):
    """Accuracy yardstick: the max norm-relative error against the exact scaled product is of the size of the fp8 backends'
    (cuBLAS / cuDNN through ``bmm_fp8``, whichever runs the format: cuBLASLt rejects e5m2 x e5m2), for the unsplit class and
    the split class (N = 128: C = 2); where no backend runs the format, the error is bounded by the output rounding."""
    A, B = adv_fp8(1, M, K, seed=11, dt=dt, cancel=False), weights(1, N, K, 12, dt)
    sa, sb = scales()
    ours = norm_rel_err(bi_bmm(A, B, sa, sb)[0], A[0], B[0], sa, sb)
    refs = []
    for backend in ("cublas", "cudnn"):
        try:
            c = flashinfer.bmm_fp8(A, B, sa, sb, torch.bfloat16, backend=backend)
        except Exception:  # noqa: BLE001  (the backend does not run this format)
            continue
        refs.append(norm_rel_err(c[0], A[0], B[0], sa, sb))
    if refs:
        assert ours <= 1.1 * min(refs) + 1e-6, (ours, refs)
    else:
        assert ours <= 2.0**-8, (
            ours
        )  # the bf16 output's rounding bounds the norm-relative error of an fp32 accumulation


@needs_device
@pytest.mark.parametrize("dt", [E4, E5], ids=["e4m3", "e5m2"])
@pytest.mark.parametrize("od", [torch.bfloat16, torch.float16], ids=["bf16", "f16"])
@pytest.mark.parametrize("b,M,N,K", [(3, 300, 4096, 4096), (2, 37, 128, 8192)])
def test_determinism_and_graph_replay(b, M, N, K, od, dt):
    """Run to run and under CUDA-graph replay, for both fp8 formats and both output dtypes: the batched unsplit launch
    (N = 4096) and the per-element split launch (N = 128, C = 4)."""
    A, B = adv_fp8(b, M, K, seed=21, dt=dt), weights(b, N, K, 22, dt=dt)
    sa, sb = scales() if od == torch.bfloat16 else scales(2**-9 * 1.3, 2**-6 * 0.7)
    ref = bits(bi_bmm(A, B, sa, sb, od)).clone()
    for _ in range(3):
        assert torch.equal(bits(bi_bmm(A, B, sa, sb, od)), ref)
    out = torch.empty(b, M, N, dtype=od, device="cuda")
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        bi_bmm(
            A, B, sa, sb, od, out=out
        )  # warm-up on the side stream: the compile happens outside the capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            bi_bmm(A, B, sa, sb, od, out=out)
    torch.cuda.current_stream().wait_stream(s)
    for _ in range(2):
        out.zero_()
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(bits(out), ref)


@needs_device
def test_batch_invariance_along_rows_and_batch():
    """R2 along both axes: a row's bits do not depend on the number of rows, its position or the other rows; a batch
    element's bits do not depend on the batch size, its position in the batch or the other elements (one element alone,
    through b = 1, equals its slice of the batched call); the dispatcher's M-dependent picks change no bit."""
    N, K = 4096, 4096
    A, B = adv_fp8(1, 2049, K, seed=31), weights(1, N, K, 32)
    sa, sb = scales()
    full = bits(bi_bmm(A, B, sa, sb))[0]
    for M in (1, 16, 37, 64, 128, 256, 512, 1000, 2049):
        assert torch.equal(
            bits(bi_bmm(A[:, :M].contiguous(), B, sa, sb))[0], full[:M]
        ), f"M={M}"
    x = torch.cat(
        [adv_fp8(1, 5, K, seed=33), A[:, 100:101], adv_fp8(1, 7, K, seed=34)], dim=1
    )
    assert torch.equal(bits(bi_bmm(x, B, sa, sb))[0, 5], full[100])
    # along the batch: five elements, different weights per element
    A5, B5 = adv_fp8(5, 64, K, seed=35), weights(5, N, K, 36)
    out5 = bits(bi_bmm(A5, B5, sa, sb))
    for i in range(5):
        alone = bits(bi_bmm(A5[i : i + 1].contiguous(), B5[i : i + 1], sa, sb))[0]
        assert torch.equal(alone, out5[i]), f"element {i}"
    sub = bits(bi_bmm(A5[1:4].contiguous(), B5[1:4], sa, sb))
    assert torch.equal(sub, out5[1:4])
    perm = torch.tensor([3, 0, 4, 1, 2], device="cuda")
    assert torch.equal(
        bits(bi_bmm(A5[perm].contiguous(), B5[perm], sa, sb)), out5[perm]
    )
    # the dispatcher's picks for other row counts and parts compute the same bits
    for M in (16, 1000):
        c1 = mm_fp8_batch_invariant(
            A[0, :M].clone(),
            B[0],
            sa,
            sb,
            config=default_config(M, N, K, 1, "k", 204, "sm_107", "e4m3"),
        )
        c2 = mm_fp8_batch_invariant(
            A[0, :M].clone(),
            B[0],
            sa,
            sb,
            config=default_config(2049, N, K, 1, "k", 148, "sm_100", "e4m3"),
        )
        assert torch.equal(bits(c1), bits(c2)) and torch.equal(bits(c1), full[:M])


@needs_device
@pytest.mark.parametrize("N,K", [(2048, 4096), (128, 8192)])
def test_permutation_and_subbatch_invariance(N, K):
    """R2 in both forms along the rows, for the unsplit (N = 2048) and the split (N = 128, C = 4) class."""
    A, B = adv_fp8(1, 512, K, seed=41), weights(1, N, K, 42)
    sa, sb = scales()
    full = bits(bi_bmm(A, B, sa, sb))[0]
    perm = torch.randperm(512, generator=torch.Generator().manual_seed(1)).cuda()
    assert torch.equal(bits(bi_bmm(A[:, perm].contiguous(), B, sa, sb))[0], full[perm])
    for k in (1, 2, 4, 8, 16, 37):
        idx = torch.randperm(512, generator=torch.Generator().manual_seed(k))[:k].cuda()
        assert torch.equal(
            bits(bi_bmm(A[:, idx].contiguous(), B, sa, sb))[0], full[idx]
        ), f"k={k}"


@needs_device
def test_weight_layouts_agree():
    """R4: the column-major (b, K, N) weight (the transpose of a row-major (b, N, K) tensor, ``bmm_fp8``'s documented
    layout) and the row-major (b, K, N) weight give the same bits, through the API and the one-product entry; M = 1 with a
    non-contiguous row too."""
    A, B = adv_fp8(2, 300, 4000, seed=51), weights(2, 1024, 4000, 52)
    sa, sb = scales()
    c_k = bi_bmm(A, B, sa, sb)
    c_n = bi_bmm(A, B.contiguous(), sa, sb)
    assert B.stride(1) == 1 and B.contiguous().stride(2) == 1
    assert torch.equal(bits(c_k), bits(c_n))
    row = A[0, 7:8, :]
    assert torch.equal(
        bits(mm_fp8_batch_invariant(row, B[0], sa, sb)), bits(c_k[0, 7:8])
    ) and torch.equal(
        bits(mm_fp8_batch_invariant(row, B[0].contiguous(), sa, sb)), bits(c_k[0, 7:8])
    )


@needs_device
@pytest.mark.parametrize("M,N,K,dt", [(1000, 4096, 4096, E4), (37, 2048, 4000, E5)])
def test_config_agreement(M, N, K, dt):
    """R3: every configuration the tuner may choose (the dispatcher's top candidates on every part, both layouts) and a spread
    of hand-picked records (64- / 128- / 256-row tiles, 1- and 2-CTA MMA, clusters, both rasters, swizzles, the 16-wide decode
    tile) give identical bits, for both weight layouts."""
    fmt = "e4m3" if dt == E4 else "e5m2"
    a, Bk = adv_fp8(M, K, seed=61, dt=dt), weights(1, N, K, 62, dt)[0]
    Bn = Bk.contiguous()
    sa, sb = scales()
    cfgs = []
    for family, sm in PARTS:
        for lay in ("k", "n"):
            cfgs += rank_topk(M, N, K, 1, lay, sm, 10, family, fmt)
    cfgs += [
        Knobs((64, 32), (1, 1), False, True, "m", 1),
        Knobs((128, 128), (1, 2), False, True, "n", 2),
        Knobs((128, 256), (2, 1), True, True, "m", 1),
        Knobs((256, 64), (2, 2), True, True, "n", 4),
        Knobs((128, 16), (1, 1), False, True, "m", 1),
        Knobs((128, 16), (2, 1), True, True, "m", 1),
        Knobs((64, 192), (4, 1), False, True, "m", 8),
    ]
    cfgs = list(dict.fromkeys(cfgs))
    ref = None
    checked = 0
    for lay, B in (("k", Bk), ("n", Bn)):
        for c in cfgs:
            if not can_implement(c, M, N, K, lay, 1, fmt):
                continue
            got = bits(mm_fp8_batch_invariant(a, B, sa, sb, config=c))
            if ref is None:
                ref = got
            assert torch.equal(got, ref), f"{c.describe()} ({lay}) differs"
            checked += 1
    assert checked >= 24


@needs_device
@pytest.mark.parametrize("dt", [E4, E5], ids=["e4m3", "e5m2"])
@pytest.mark.parametrize(
    "M,N,K,od", [(37, 128, 8192, torch.bfloat16), (300, 64, 16384, torch.float16)]
)
def test_cta_chunk_fold_equals_workspace_split(M, N, K, od, dt):
    """The split class ``kchunk[C].scaled`` has two implementations: the workspace split (chunks on different CTAs, an fp32
    workspace, a fixup fold in chunk order and the chain) and the in-CTA chunk fold (one CTA folds every chunk of its tile
    through a TMEM running sum, the chain after the last chunk). They are one term in Lean (``ctaSplit32_eq_kchunk32``)
    and one bit pattern here, on both weight layouts and both output dtypes; and the split class is a different
    specification from the unsplit fold (expected)."""
    fmt = "e4m3" if dt == E4 else "e5m2"
    C = chunks_for(N, K, fmt)
    assert C > 1
    a, Bk = adv_fp8(M, K, seed=71, dt=dt), weights(1, N, K, 72, dt=dt)[0]
    sa, sb = scales() if od == torch.bfloat16 else scales(2**-9 * 1.3, 2**-6 * 0.7)
    ws = Knobs((128, 64), (1, 1), False, True, "m", 1)
    ref = bits(mm_fp8_batch_invariant(a, Bk, sa, sb, out_dtype=od, config=ws, chunks=C))
    recs = [
        k for k in rank_topk(M, N, K, C, "k", 204, 10, "sm_107", fmt) if k.cta_chunks
    ]
    recs += [
        Knobs((64, 32), (1, 4), False, True, "m", 1, 0, True),
        Knobs((128, 64), (1, 1), False, True, "n", 1, 0, True),
        Knobs((256, 128), (2, 1), True, True, "m", 2, 0, True),
    ]
    recs = [k for k in dict.fromkeys(recs) if can_implement(k, M, N, K, "k", C, fmt)]
    assert len(recs) >= 3
    for k in recs:
        for B in (Bk, Bk.contiguous()):
            assert torch.equal(
                bits(
                    mm_fp8_batch_invariant(
                        a, B, sa, sb, out_dtype=od, config=k, chunks=C
                    )
                ),
                ref,
            ), k.describe()
    assert torch.equal(
        bits(bi_bmm(a.unsqueeze(0), Bk.unsqueeze(0), sa, sb, od))[0], ref
    )  # the dispatcher's own pick (a per-element launch of the split class) is a member too
    assert not torch.equal(
        bits(mm_fp8_batch_invariant(a, Bk, sa, sb, out_dtype=od, config=ws, chunks=1)),
        ref,
    )


@needs_device
def test_scale_chain_is_exact():
    """``out == cvt_rn(mul_rn(acc, mul_rn(a_scale, b_scale)))`` bit for bit, where ``acc`` is the kernel's own fp32 accumulator
    (the fp32 output under unit scales: x * 1.0 is exact). Both CPU fp32 multiplies are correctly rounded, as is the bf16 /
    fp16 conversion; the unsplit class and the split class (the fixup's chain and the in-CTA fold's) alike."""
    one = torch.ones(1, device="cuda")
    for (M, N, K), od in [
        ((300, 2048, 1024), torch.bfloat16),
        ((64, 4096, 4096), torch.float16),
        ((16, 64, 65536), torch.bfloat16),
        ((16, 64, 65536), torch.float16),
    ]:
        a, B = adv_fp8(M, K, seed=81), weights(1, N, K, 82)[0]
        sa, sb = scales()
        acc = mm_fp8_batch_invariant(a, B, one, one, out_dtype=torch.float32).cpu()
        exact = (acc * (sa.cpu() * sb.cpu())).to(od)
        got = mm_fp8_batch_invariant(a, B, sa, sb, out_dtype=od).cpu()
        assert torch.equal(bits(got), bits(exact)), (
            M,
            N,
            K,
            od,
            chunks_for(N, K, "e4m3"),
        )
        if chunks_for(N, K, "e4m3") > 1:
            cc = Knobs((128, 64), (1, 1), False, True, "m", 1, 0, True)
            got_cc = mm_fp8_batch_invariant(a, B, sa, sb, out_dtype=od, config=cc).cpu()
            assert torch.equal(bits(got_cc), bits(exact))


@needs_device
def test_workspace_split_at_large_m():
    """The split class with its partials in the workspace at M beyond the hardware's grid.y limit (the fixup kernel strides
    over rows): bit-equal to the in-CTA chunk fold and to its own sub-batch; a caller's workspace is checked."""
    M, N, K = 70000, 128, 8192
    C = chunks_for(N, K, "e4m3")
    assert C == 4
    a, B = adv_fp8(M, K, seed=111, cancel=False), weights(1, N, K, 112)[0]
    sa, sb = scales()
    ws_rec = Knobs((128, 64), (1, 1), False, True, "m", 1)
    cta_rec = Knobs((128, 64), (1, 1), False, True, "m", 1, 0, True)
    full = bits(mm_fp8_batch_invariant(a, B, sa, sb, config=ws_rec, chunks=C))
    assert torch.equal(
        full, bits(mm_fp8_batch_invariant(a, B, sa, sb, config=cta_rec, chunks=C))
    )
    assert torch.equal(
        bits(mm_fp8_batch_invariant(a[:1000], B, sa, sb, config=ws_rec, chunks=C)),
        full[:1000],
    )
    ws = torch.empty(C * M * N + 4, dtype=torch.float32, device="cuda")
    assert torch.equal(
        bits(
            mm_fp8_batch_invariant(
                a, B, sa, sb, config=ws_rec, chunks=C, workspace=ws[: C * M * N]
            )
        ),
        full,
    )
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="workspace"):
        mm_fp8_batch_invariant(
            a[:16], B, sa, sb, config=ws_rec, chunks=C, workspace=ws[1 : 1 + C * 16 * N]
        )  # 4-byte offset


@needs_device
def test_persistent_grid_size_does_not_change_bits():
    """R2 for the SM count: a smaller persistent grid is what a smaller part, an SM carveout, MPS or a green context
    produce; it only decides which CTA computes which tile (and, for the split class, which chunk)."""
    sa, sb = scales()
    for M, N, K in [(1000, 4096, 4096), (64, 64, 65536)]:
        a, B = adv_fp8(M, K, seed=91), weights(1, N, K, 92)[0]
        C = chunks_for(N, K, "e4m3")
        base = default_config(M, N, K, C, "k", 204, "sm_107", "e4m3")
        ref = bits(mm_fp8_batch_invariant(a, B, sa, sb, config=base, chunks=C))
        for g in (1, 3, 17, 74):
            cfg = dataclasses.replace(base, grid_clusters=g)
            assert torch.equal(
                bits(mm_fp8_batch_invariant(a, B, sa, sb, config=cfg, chunks=C)), ref
            ), f"grid={g} at {(M, N, K)}"


@needs_device
@pytest.mark.parametrize("b,N,K,layout", [(1, 4096, 4096, "k"), (4, 64, 8192, "n")])
def test_autotuning_profiles_every_candidate_and_keeps_the_bits(b, N, K, layout):
    """Under ``autotune(True)`` the FlashInfer autotuner profiles the class's candidates (none may be dropped as
    unsupported) and the tuned process computes the same bits as the untuned one, during and after tuning."""
    A = adv_fp8(b, 300, K, seed=101)
    B = weights(b, N, K, 102)
    B = B if layout == "k" else B.contiguous()
    sa, sb = scales()
    ref = [bits(bi_bmm(A[:, :M].contiguous(), B, sa, sb)) for M in (1, 16, 300)]
    tuner = AutoTuner.get()
    before = {
        k: len(v)
        for k, v in tuner.stats.failed_tactics.items()
        if tuning.CUSTOM_OP_BMM_FP8 in k
    }
    with autotune(True):
        tuned = [bits(bi_bmm(A[:, :M].contiguous(), B, sa, sb)) for M in (1, 16, 300)]
    after = {
        k: len(v)
        for k, v in tuner.stats.failed_tactics.items()
        if tuning.CUSTOM_OP_BMM_FP8 in k
    }
    assert after == before, f"tactics failed while profiling: {after}"
    post = [bits(bi_bmm(A[:, :M].contiguous(), B, sa, sb)) for M in (1, 16, 300)]
    for r, t, p in zip(ref, tuned, post, strict=True):
        assert torch.equal(r, t) and torch.equal(r, p)
    assert torch.equal(post[1], post[2][:, :16])


@needs_device
@pytest.mark.parametrize("N", [24, 40, 136])
def test_n_a_multiple_of_8_with_a_k_major_weight(N):
    """The typed checks require N as a multiple of 8 (the output's 16-byte rows) and a multiple of 16 only for an N-major B
    (its 16-byte rows); the compiled kernel declares the same, so these shapes run (the split class at N = 24 / 40, the
    unsplit at N = 136) through the one-product and the batched entries, and the N-major layout still raises."""
    M, K = 16, 4096
    a, Bk = adv_fp8(M, K, seed=161), weights(1, N, K, 162)[0]
    sa, sb = scales()
    out = mm_fp8_batch_invariant(a, Bk, sa, sb)
    assert out.shape == (M, N) and norm_rel_err(out, a, Bk, sa, sb) <= 2.0**-8
    batched = bmm_fp8_batch_invariant(a[None], Bk[None], sa, sb, torch.bfloat16)
    assert torch.equal(bits(batched[0]), bits(out))
    assert torch.equal(bits(bi_bmm(a[None], Bk[None], sa, sb)), bits(batched))
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="multiple of 16"):
        mm_fp8_batch_invariant(a, Bk.contiguous(), sa, sb)


@needs_device
def test_trace_apply_does_not_replace_the_class():
    """A Trace Apply solution registered for the call's definition name replaces the plain API call (the control); under
    ``batch_invariant=True`` it must not: the flag fixes the call's dataflow to the class, and the wrapper passes the call
    through to the operator, whose flag path runs the class."""
    import flashinfer.trace_apply as ta
    from flashinfer.trace_apply.apply import (
        _registry_by_fi_api,
        bind_namespace,
        build_extractor_maps,
        extract_axes,
    )

    b, M, N, K = 1, 16, 4096, 4096
    A, B = adv_fp8(b, M, K, seed=171), weights(b, N, K, 172)
    sa, sb = scales()
    ref = bits(bi_bmm(A, B, sa, sb)).clone()
    reg = _registry_by_fi_api()
    key = next(k for k in reg if k.endswith(".bmm_fp8"))
    original, templates = reg[key]
    ns = bind_namespace(original, (A, B, sa, sb, torch.bfloat16), {})
    name = templates[0].definition_name(
        extract_axes(build_extractor_maps(templates), ns)
    )
    calls: list = []

    def solution(*args, **kwargs):
        calls.append(1)
        return torch.full((b, M, N), 2.0, dtype=torch.bfloat16, device="cuda")

    ta.enable_apply({name: solution})
    try:
        plain = flashinfer.bmm_fp8(A, B, sa, sb, torch.bfloat16)
        assert calls and bool(torch.all(plain == 2.0)), (
            "the control: the solution replaces the plain call"
        )
        seen = len(calls)
        got = flashinfer.bmm_fp8(
            A, B, sa, sb, torch.bfloat16, backend="auto", batch_invariant=True
        )
        assert len(calls) == seen, (
            "the registered solution ran under batch_invariant=True"
        )
        assert torch.equal(bits(got), ref)
        assert ta.stats().get(key, {}).get("fallback_batch_invariant", 0) >= 1
    finally:
        ta.disable_apply()


@needs_device
def test_public_api_skips_backend_selection(monkeypatch):
    """batch_invariant=True never runs a backend, so the @backend_requirement wrapper must not run the auto selection
    for it (host cost only): a selecting call rewrites `bmm_fp8.suitable_auto_backends`, a bypassed one leaves it."""
    A, B = adv_fp8(2, 16, 4096, seed=131), weights(2, 4096, 4096, 132)
    sa, sb = scales()
    sentinel = ["sentinel"]
    # the wrapper creates the attribute at its first auto selection, so it may not exist yet
    monkeypatch.setattr(
        flashinfer.bmm_fp8, "suitable_auto_backends", sentinel, raising=False
    )
    bi_bmm(A, B, sa, sb)
    assert flashinfer.bmm_fp8.suitable_auto_backends is sentinel


@needs_device
def test_unsupported_inputs_raise_and_nothing_falls_back():
    A, B = adv_fp8(2, 16, 4096, seed=121), weights(2, 4096, 4096, 122)
    sa, sb = scales()
    with pytest.raises(bi.BatchInvariantUnsupportedError, match='backend="auto"'):
        flashinfer.bmm_fp8(
            A, B, sa, sb, torch.bfloat16, backend="cublas", batch_invariant=True
        )
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="both float8"):
        bi_bmm(A, weights(2, 4096, 4096, 123, E5), sa, sb)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="1-element float32"):
        bi_bmm(A, B, torch.ones(2, device="cuda"), sb)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="1-element float32"):
        bi_bmm(A, B, sa, sb.to(torch.bfloat16))
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="on A's device"):
        bi_bmm(A, B, sa, sb.cpu())
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="multiple of 16"):
        bi_bmm(adv_fp8(2, 16, 4104, seed=124), weights(2, 64, 4104, 125), sa, sb)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="matching batch"):
        bi_bmm(A, weights(3, 4096, 4096, 126), sa, sb)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="3-D operands"):
        bmm_fp8_batch_invariant(A[0], B[0], sa, sb, torch.bfloat16)
    # the public API hands batch-invariant calls to the class without running the backend checks: the class's typed
    # error (a ValueError, as before)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="bf16 or fp16 output"):
        bi_bmm(A, B, sa, sb, torch.float32)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="out"):
        bi_bmm(
            A,
            B,
            sa,
            sb,
            out=torch.empty(2, 16, 4096, dtype=torch.float16, device="cuda"),
        )
    # nothing to launch: no rows, or no batch elements
    out = bi_bmm(A[:, :0], B, sa, sb)
    assert out.shape == (2, 0, 4096) and out.dtype == torch.bfloat16
    out = bi_bmm(A[:0], B[:0], sa, sb, torch.float16)
    assert out.shape == (0, 16, 4096) and out.dtype == torch.float16


@needs_device
def test_chunk_count_and_workspace_checks():
    """``chunks`` must be positive (the typed error, not a division by zero), and a caller workspace may not overlap the
    output or a scale: the split class writes its partials there before the fixup reads them."""
    b, M, N, K = (
        3,
        64,
        128,
        8192,
    )  # N = 128: the chunk rule splits; b = 3: one workspace reused per element
    A, B = adv_fp8(b, M, K, seed=131), weights(b, N, K, 132)
    sa, sb = scales()
    # a caller-forced configuration the kernel cannot implement: the 16-wide tile of the 2-CTA pair needs a K-major weight
    # on kind::f8f6f4 (the only guard is the launch path's, for the unsplit class the batched launch's)
    A1, B1 = adv_fp8(1, 16, 4096, seed=141), weights(1, 4096, 4096, 142)
    forced = Knobs((256, 16), (2, 1), True, True, "m", 1)
    assert not can_implement(forced, 16, 4096, 4096, "n", 1, "e4m3")
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="enumerated space"):
        bmm_fp8_batch_invariant(
            A1, B1.contiguous(), sa, sb, torch.bfloat16, config=forced
        )
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="chunks = 0"):
        bmm_fp8_batch_invariant(A, B, sa, sb, torch.bfloat16, chunks=0)
    C = chunks_for(N, K, "e4m3")
    assert C > 1
    ws = torch.empty(C * M * N + 64, dtype=torch.float32, device="cuda")
    ref = bits(bi_bmm(A, B, sa, sb)).clone()
    got = bmm_fp8_batch_invariant(
        A, B, sa, sb, torch.bfloat16, workspace=ws[: C * M * N]
    )
    assert torch.equal(bits(got), ref)
    for i in range(
        b
    ):  # every element through the one reused workspace equals its own single-element call
        single = bmm_fp8_batch_invariant(
            A[i : i + 1], B[i : i + 1], sa, sb, torch.bfloat16
        )
        assert torch.equal(bits(got[i]), bits(single[0])), i
    # the output may not alias an input or a scale on either launch path: the bits would depend on which CTA runs first
    buf = torch.empty(M, 2 * N, dtype=torch.uint8, device="cuda")
    out_alias = buf.view(
        torch.bfloat16
    )  # (M, N) bf16 over the same bytes as the (M, K <= 2N) fp8 operand below
    a_alias = buf[:, : min(K, 2 * N)].view(torch.float8_e4m3fn)
    Ka = a_alias.shape[1]
    w_alias = weights(1, N, Ka, 133)[0]
    with pytest.raises(
        bi.BatchInvariantUnsupportedError, match="A overlapping the output"
    ):
        mm_fp8_batch_invariant(a_alias, w_alias, sa, sb, out=out_alias)
    a2 = adv_fp8(M, 4096, seed=134)
    w_buf = torch.empty(N, 4096, dtype=torch.float8_e4m3fn, device="cuda")
    out_w = (
        w_buf.view(torch.uint8).view(-1)[: 2 * M * N].view(torch.bfloat16).view(M, N)
    )  # a contiguous output over the weight's first rows
    with pytest.raises(
        bi.BatchInvariantUnsupportedError, match="B overlapping the output"
    ):
        mm_fp8_batch_invariant(a2, w_buf.t(), sa, sb, out=out_w)
    out_s = torch.empty(M, 4096, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(
        bi.BatchInvariantUnsupportedError, match="a scale overlapping the output"
    ):
        mm_fp8_batch_invariant(
            a2,
            weights(1, 4096, 4096, 135)[0],
            out_s.view(torch.float32)[0, :1],
            sb,
            out=out_s,
        )
    with pytest.raises(
        bi.BatchInvariantUnsupportedError, match="overlapping the workspace"
    ):
        bmm_fp8_batch_invariant(A, B, ws[:1], sb, torch.bfloat16, workspace=ws)
    big = torch.empty(
        b * 16 * K // 4, dtype=torch.float32, device="cuda"
    )  # a workspace holding the (b, 16, K) operand below
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="A overlapping"):
        bmm_fp8_batch_invariant(
            big.view(torch.float8_e4m3fn).view(b, 16, K),
            B,
            sa,
            sb,
            torch.bfloat16,
            workspace=big,
        )
    out = ws.view(torch.bfloat16)[: b * M * N].view(b, M, N)
    with pytest.raises(
        bi.BatchInvariantUnsupportedError, match="overlapping the workspace"
    ):
        bmm_fp8_batch_invariant(A, B, sa, sb, torch.bfloat16, out=out, workspace=ws)
    out = torch.empty(b, M, N, dtype=torch.bfloat16, device="cuda")
    got = bmm_fp8_batch_invariant(A, B, sa, sb, torch.bfloat16, out=out, workspace=ws)
    assert torch.equal(bits(got), ref)


@needs_device
def test_singleton_row_output_takes_any_row_stride():
    """At M = 1 the output's row stride is a size-1 mode's: a (b, 1, N) output with strides (N, 1, 1) gets the bits of a
    contiguous one."""
    b, N, K = 2, 2048, 4096
    A, B = adv_fp8(b, 1, K, seed=141), weights(b, N, K, 142)
    sa, sb = scales()
    ref = bits(bi_bmm(A, B, sa, sb)).clone()
    out = torch.empty(b, N, 1, dtype=torch.bfloat16, device="cuda").permute(0, 2, 1)
    assert out.stride() == (N, 1, 1)
    assert torch.equal(bits(bi_bmm(A, B, sa, sb, out=out)), ref)
