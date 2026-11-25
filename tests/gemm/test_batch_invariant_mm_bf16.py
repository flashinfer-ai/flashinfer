# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Batch-invariant BF16 GEMM: numeric reference, row invariance, layouts and CUDA Graph replay."""

from unittest.mock import patch

import pytest
import torch

import flashinfer
from flashinfer import batch_invariant as bi
from flashinfer.gemm import gemm_base
from flashinfer.gemm.batch_invariant.api import (
    mm_bf16_batch_invariant,
)
from flashinfer.gemm.batch_invariant.blocks.operands import can_implement
from flashinfer.gemm.batch_invariant.spec import spec_fingerprint
from flashinfer.gemm.batch_invariant.heuristic import PARTS
from flashinfer.gemm.batch_invariant.generator import (
    REFERENCE_SHAPE,
    Knobs,
    enumerate_space,
)
from flashinfer.gemm.batch_invariant.schedule import (
    chunks_for,
    default_config,
    rank_topk,
)

needs_device = pytest.mark.usefixtures("cudnn_batch_invariant_device")


def bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16)


def adv_bf16(rows: int, cols: int, seed: int) -> torch.Tensor:
    """Adversarial bf16 operand: exponents spread over +-2^20, near-cancelling neighbour pairs, 2^12 outliers, zeros
    of both signs and subnormals, so that any accumulation-order difference reaches the output bits."""
    g = torch.Generator().manual_seed(seed)
    e = (torch.rand(rows, cols, generator=g) * 2 - 1) * 20
    x = (torch.rand(rows, cols, generator=g) * 2 - 1) * torch.exp2(e)
    k = torch.arange(0, cols - 1, 8)
    x[:, k + 1] = -x[:, k] * (1 + 2.0**-7)
    x[torch.randint(0, 64, (rows, cols), generator=g) == 0] *= 2.0**12
    z = torch.randint(0, 97, (rows, cols), generator=g)
    x[z == 0] = 0.0
    x[z == 1] = -0.0
    x[z == 2] = 2.0**-130
    return x.to(torch.bfloat16).cuda()


def rnd_bf16(rows: int, cols: int, seed: int, span: float = 6.0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    e = (torch.rand(rows, cols, generator=g) * 2 - 1) * span
    return (
        ((torch.rand(rows, cols, generator=g) * 2 - 1) * torch.exp2(e))
        .to(torch.bfloat16)
        .cuda()
    )


def norm_rel_err(c: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> float:
    ref = a.double() @ b.double()
    scale = a.double().abs() @ b.double().abs()
    return ((c.double() - ref).abs() / (scale + 1e-30)).max().item()


def int_bf16(rows: int, cols: int, seed: int, span: int = 4) -> torch.Tensor:
    """Integer-valued bf16 operand in [-span, span]: every product and every fp32 partial sum of a GEMM over it is an
    exact integer (far below 2^24), whatever the accumulation order."""
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randint(-span, span + 1, (rows, cols), generator=g)
        .to(torch.bfloat16)
        .cuda()
    )


def frac_bias(n: int, seed: int) -> torch.Tensor:
    """A bf16 bias of eighths in (-32, 32): exact in bf16, and its fp32 sum with an integer accumulator below 2^16 is
    exact too, so the class's single rounding (after the fp32 add) is the only rounding of the biased chain."""
    g = torch.Generator().manual_seed(seed)
    return (torch.randint(-255, 256, (n,), generator=g) / 8.0).to(torch.bfloat16).cuda()


def exact_biased_reference(
    a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """``cvt_bf16(add_f32(acc, bias[n]))`` computed exactly: the fp64 product is the exact integer accumulator, the sum
    with the bias is exact in fp32 (``.float()``), ``.to(bf16)`` is the one round-to-nearest-even."""
    return (a.double() @ b.double() + bias.double()).float().to(torch.bfloat16)


def bi_mm(a, b, **kw):
    return flashinfer.mm_bf16(a, b, backend="auto", batch_invariant=True, **kw)


def k_major(b: torch.Tensor) -> torch.Tensor:
    """The same (K, N) weight as the transpose of a row-major (N, K) tensor."""
    return b.t().contiguous().t()


# ----------------------------------------------------------------------------------------------- device-free


def test_fingerprint_is_a_function_of_the_weight_shape():
    f = bi.fingerprint("mm_bf16", N=4096, K=4096)
    assert f.startswith("gemm-bf16-kfold-") and f == bi.fingerprint(
        "mm_bf16", N=4096, K=4096
    )
    assert bi.fingerprint("mm_bf16", N=128, K=4096).startswith("gemm-bf16-kchunk2-")
    assert bi.fingerprint("mm_bf16", N=64, K=8192).startswith("gemm-bf16-kchunk8-")
    assert bi.fingerprint("mm_bf16", N=0, K=4096).startswith("gemm-bf16-kfold-")
    assert bi.fingerprint("mm_bf16", N=4096, K=4096) != bi.fingerprint(
        "mm_bf16", N=4096, K=4096 + 16
    )
    assert bi.class_name("mm_bf16") == "gemm.bf16"
    with pytest.raises(TypeError):
        bi.fingerprint("mm_bf16", N=4096)
    with pytest.raises(TypeError):
        bi.fingerprint("mm_bf16", N=4096, K=4096, M=16)
    # the bias is an optional dimension: a biased call is a different specification, bias=False the unbiased one
    fb = bi.fingerprint("mm_bf16", N=4096, K=4096, bias=True)
    with pytest.raises(
        ValueError, match="bf16 class only"
    ):  # no biased fp8 specification exists
        spec_fingerprint(4096, "bf16", 1, "e4m3", bias=True)
    assert fb.startswith("gemm-bf16-kfold-bias-") and fb != f
    assert bi.fingerprint("mm_bf16", N=4096, K=4096, bias=False) == f
    assert bi.fingerprint("mm_bf16", N=128, K=4096, bias=True).startswith(
        "gemm-bf16-kchunk2-bias-"
    )


def test_unbiased_fingerprints_are_unchanged():
    """The unbiased class's identifiers are pinned: adding the bias dimension (or anything else) must not move the
    bits, hence not the fingerprint, of a bias-free call."""
    assert (
        bi.fingerprint("mm_bf16", N=4096, K=4096) == "gemm-bf16-kfold-aae1814611831992"
    )
    assert (
        bi.fingerprint("mm_bf16", N=128, K=4096) == "gemm-bf16-kchunk2-287033baede9fe6e"
    )
    assert (
        bi.fingerprint("mm_bf16", N=64, K=8192) == "gemm-bf16-kchunk8-caf0be5a5d78f061"
    )
    assert (
        bi.fingerprint("mm_bf16", N=512, K=16384)
        == "gemm-bf16-kchunk2-a9820c23c042f615"
    )


def test_biased_chain_adds_the_bias_once_after_the_fold():
    """The specification's term: one ``add`` of the bias leaf of the output column around the whole fold, then the
    single conversion; the fold itself is the unbiased class's."""
    from flashinfer.batch_invariant.fp_interface import term
    from flashinfer.gemm.batch_invariant.spec import GemmSpec

    for K, C in [(64, 1), (256, 2)]:
        biased = term.sexpr(GemmSpec(K, chunks=C, bias=True).out(0, 3))
        plain = term.sexpr(GemmSpec(K, chunks=C).out(0, 3))
        assert biased.count("(bias 3)") == 1 and "(bias" not in plain
        acc = plain[len("(cvt bf16 ") : -1]
        assert biased in (
            f"(cvt bf16 (add (bias 3) {acc}))",
            f"(cvt bf16 (add {acc} (bias 3)))",
        )


def test_chunk_rule_takes_no_m():
    import inspect

    params = list(inspect.signature(chunks_for).parameters)
    assert params == ["N", "K", "in_fmt"] and "M" not in params
    assert chunks_for(4096, 4096) == 1 and chunks_for(8192, 8192) == 1
    assert chunks_for(0, 4096) == 1  # an empty weight: no split, as in Lean
    assert (
        chunks_for(128, 4096) == 2
        and chunks_for(64, 8192) == 8
        and chunks_for(64, 262144) == 128
    )
    for N, K in [(64, 64), (512, 65536), (2048, 2048)]:
        C = chunks_for(N, K)
        assert C & (C - 1) == 0 and K % (16 * C) == 0 and (C == 1 or K // C >= 256)


@pytest.mark.parametrize("family,sm_count", PARTS)
def test_dispatcher_picks_are_members_of_the_space(family, sm_count):
    """R3 membership: every configuration the dispatcher or the tuner's candidate list returns is a well-formed record
    the kernel accepts for the shape; the candidates are distinct and the default is their first."""
    space = set(enumerate_space(*REFERENCE_SHAPE))
    for M in (1, 16, 300, 4096):
        for N, K in [
            (4096, 4096),
            (14336, 4096),
            (128, 4096),
            (64, 8192),
            (2048, 16384),
        ]:
            C = chunks_for(N, K)
            for b_major in ("n", "k"):
                top = rank_topk(M, N, K, C, b_major, sm_count, 10, family)
                assert 1 <= len(top) <= 10 and len(set(top)) == len(top)
                assert top[0] == default_config(M, N, K, C, b_major, sm_count, family)
                for cfg in top:
                    assert cfg.well_formed() and can_implement(
                        cfg, M, N, K, b_major, C
                    ), cfg.describe()
                    if not cfg.cta_chunks:
                        assert cfg in space, cfg.describe()


@pytest.mark.parametrize("family,sm_count", PARTS)
def test_memoised_dispatcher_pick_equals_a_fresh_ranking(family, sm_count):
    """The dispatcher's pick, its candidate list and the enumerated space are memoised per argument tuple: a memoised
    answer is the fresh computation's, element for element, over a grid of shapes of every format; the memos are
    cleared with the rankings (a device's registered numbers, ``rank.cache_clear()``), and a memoised list is never the
    caller's to mutate into the memo."""
    from flashinfer.gemm.batch_invariant import heuristic as H
    from flashinfer.gemm.batch_invariant import schedule as S
    from flashinfer.gemm.batch_invariant.generator import _enumerate_space_cached

    cases = []
    for in_fmt, layouts in [
        ("bf16", ("n", "k")),
        ("e4m3", ("n", "k")),
        ("mxfp8", ("k",)),
        ("nvfp4", ("k",)),
        ("mxfp4w8a", ("k",)),
        ("e4m3_bs", ("k",)),
    ]:
        for N, K in [(4096, 4096), (128, 8192), (2048, 16384)]:
            C = chunks_for(N, K, in_fmt)
            for M in (1, 16, 300, 4096):
                for b_major in layouts:
                    cases.append((M, N, K, C, b_major, sm_count, family, in_fmt))
    for args in cases:
        memo = default_config(*args)
        assert memo == default_config.__wrapped__(*args), args
        assert memo == default_config(*args)  # the hit
        top = rank_topk(*args[:6], 10, *args[6:])
        fresh = S._rank_topk_cached.__wrapped__(*args[:6], 10, *args[6:], False)
        assert tuple(top) == fresh and top[0] == memo, args
        top.append(memo)  # the caller's list, not the memo
        assert len(rank_topk(*args[:6], 10, *args[6:])) == len(fresh)
    M, N, K, C, b_major, _, _, in_fmt = cases[0]
    space = enumerate_space(M, N, K, chunks=C, b_major=b_major, in_fmt=in_fmt)
    assert tuple(space) == _enumerate_space_cached.__wrapped__(
        M, N, K, S.SWIZZLES, S.CLUSTERS, C, b_major, in_fmt
    )
    space.clear()
    assert enumerate_space(M, N, K, chunks=C, b_major=b_major, in_fmt=in_fmt)
    assert default_config.cache_info().currsize >= len(cases) // 2
    H.rank.cache_clear()  # the rankings' clear takes the memos with it
    assert default_config.cache_info().currsize == 0
    assert S._rank_topk_cached.cache_info().currsize == 0
    default_config(*cases[0])
    saved = {k: dict(v) for k, v in H._QUERIED_CAPACITY.items()}
    try:
        cap = H.capacity_for(family, sm_count)
        H.register_capacity(family, sm_count, 2, cap[2] + 1)  # a changed device number
        assert default_config.cache_info().currsize == 0
    finally:
        H._QUERIED_CAPACITY.clear()
        H._QUERIED_CAPACITY.update(saved)
        H.rank.cache_clear()


def test_hbm_traffic_follows_the_tile_order():
    """The traffic model: the operands' own bytes when every band's panels stay resident in the L2, refetches per band
    or per wave when they do not, fewer of them with a wider swizzle band, and never below the operands' own bytes."""
    from flashinfer.gemm.batch_invariant.heuristic import hbm_traffic

    M, N, K, eb = 32768, 12288, 8192, 2.0
    unique = (M + N) * K * eb + M * N * 2
    args = (M, N, K, eb, 2, 256, 256, M // 256, N // 256, 1, 102)
    assert hbm_traffic(*args, 1, "m", float("inf")) == unique
    l2 = 80 * 2**20
    assert hbm_traffic(*args, 1, "m", l2) > hbm_traffic(*args, 8, "m", l2) >= unique
    assert hbm_traffic(*args, 1, "n", l2) > hbm_traffic(*args, 8, "n", l2) >= unique


def test_unsupported_device_raises_before_any_kernel():
    a = torch.zeros(4, 64, dtype=torch.bfloat16)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="gemm.bf16"):
        mm_bf16_batch_invariant(a, torch.zeros(64, 64, dtype=torch.bfloat16))


# ----------------------------------------------------------------------------------------------- on the device


@needs_device
@pytest.mark.parametrize(
    "M,N,K", [(1000, 4096, 4096), (512, 4000, 1024), (300, 128, 4096), (64, 64, 8192)]
)
def test_matches_reference(M, N, K):
    """Accuracy yardstick: the max norm-relative error against the exact product is of the size of torch.matmul's
    (cuBLAS, fp32 accumulate), for the unsplit class and the split classes (C = 2, 8)."""
    a, b = rnd_bf16(M, K, 11), rnd_bf16(K, N, 12)
    ours, ref = norm_rel_err(bi_mm(a, b), a, b), norm_rel_err(torch.matmul(a, b), a, b)
    assert ours <= 1.1 * ref + 1e-6, (ours, ref)


@needs_device
@pytest.mark.parametrize(
    "M,N,K,biased",
    [(1000, 4096, 4096, False), (37, 128, 8192, False), (1000, 4096, 4096, True)],
)
def test_determinism_and_graph_replay(M, N, K, biased):
    a, b = adv_bf16(M, K, 21), adv_bf16(K, N, 22)
    kw = {"bias": adv_bf16(1, N, 23)[0]} if biased else {}
    ref = bits(bi_mm(a, b, **kw)).clone()
    for _ in range(3):
        assert torch.equal(bits(bi_mm(a, b, **kw)), ref)
    out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        bi_mm(
            a, b, out=out, **kw
        )  # warm-up on the side stream: the compile happens outside the capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            bi_mm(a, b, out=out, **kw)
    torch.cuda.current_stream().wait_stream(s)
    for _ in range(2):
        out.zero_()
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(bits(out), ref)


@needs_device
def test_batch_invariance():
    N, K = 4096, 4096
    a, b = adv_bf16(2049, K, 31), adv_bf16(K, N, 32)
    full = bits(bi_mm(a, b))
    for M in (1, 16, 37, 64, 128, 256, 512, 1000, 2049):
        assert torch.equal(bits(bi_mm(a[:M].clone(), b)), full[:M]), f"M={M}"
    x = torch.cat([adv_bf16(5, K, 33), a[100:101], adv_bf16(7, K, 34)])
    assert torch.equal(bits(bi_mm(x, b))[5], full[100])


@needs_device
@pytest.mark.parametrize("N,K", [(2048, 4096), (128, 8192)])
def test_permutation_and_subbatch_invariance(N, K):
    """R2 in both forms, for the unsplit (N = 2048) and the split (N = 128, C = 4) class: permuting rows permutes the
    outputs row-wise, and any subset of rows reproduces exactly its rows' outputs."""
    a, b = adv_bf16(512, K, 41), adv_bf16(K, N, 42)
    full = bits(bi_mm(a, b))
    perm = torch.randperm(512, generator=torch.Generator().manual_seed(1)).cuda()
    assert torch.equal(bits(bi_mm(a[perm].contiguous(), b)), full[perm])
    for k in (1, 2, 4, 8, 16, 37):
        idx = torch.randperm(512, generator=torch.Generator().manual_seed(k))[:k].cuda()
        assert torch.equal(bits(bi_mm(a[idx].contiguous(), b)), full[idx]), f"k={k}"


@needs_device
def test_weight_layouts_agree():
    """R4: the row-major (K, N) weight and the transpose of a row-major (N, K) weight give the same bits (the class
    does not depend on B's layout; the fingerprint has no layout dimension); M = 1 with a non-contiguous row too."""
    a, b = adv_bf16(300, 4000, 51), adv_bf16(4000, 1024, 52)
    c_n, c_k = bi_mm(a, b), bi_mm(a, k_major(b))
    assert torch.equal(bits(c_n), bits(c_k))
    row = a[7:8, :]
    assert torch.equal(bits(bi_mm(row, b)), bits(c_n[7:8])) and torch.equal(
        bits(bi_mm(row, k_major(b))), bits(c_n[7:8])
    )


@needs_device
def test_public_api_skips_backend_selection(monkeypatch):
    """batch_invariant=True uses the cuDNN plan directly, so the @backend_requirement wrapper must not run the auto selection
    for it (host cost only): a selecting call rewrites `mm_bf16.suitable_auto_backends`, a bypassed one leaves it."""
    a, b = rnd_bf16(16, 4096, 121), rnd_bf16(4096, 4096, 122)
    sentinel = ["sentinel"]
    # the wrapper creates the attribute at its first auto selection, so it may not exist yet
    monkeypatch.setattr(
        flashinfer.mm_bf16, "suitable_auto_backends", sentinel, raising=False
    )
    bi_mm(a, b)
    assert flashinfer.mm_bf16.suitable_auto_backends is sentinel
    flashinfer.mm_bf16(a, b, backend="auto")  # the regular path still selects
    assert flashinfer.mm_bf16.suitable_auto_backends is not sentinel


@needs_device
def test_misaligned_a_is_copied_and_keeps_the_bits():
    # a contiguous view whose base address is two bytes past a 16-byte boundary: a placement, so the same bits
    M, K, N = 16, 4096, 4096
    a, b = rnd_bf16(M, K, 111), rnd_bf16(K, N, 112)
    buf = torch.empty(M * K + 8, dtype=torch.bfloat16, device="cuda")
    buf[1 : 1 + M * K].copy_(a.flatten())
    a_off = buf[1 : 1 + M * K].view(M, K)
    assert a_off.is_contiguous() and a_off.data_ptr() % 16 == 2
    assert torch.equal(bits(bi_mm(a_off, b)), bits(bi_mm(a, b)))
    assert torch.equal(bits(bi_mm(a_off[:1], b)), bits(bi_mm(a[:1], b)))


def test_registering_an_unchanged_device_value_keeps_the_rank_cache():
    """The device's L2 size and cluster capacities are registered on every call until the device is registered; a value
    the heuristic already holds must not clear its cached rankings (the re-ranking costs milliseconds per call)."""
    from flashinfer.gemm.batch_invariant import heuristic as H

    fam, sm = "sm_100", 148
    saved_cap = {k: dict(v) for k, v in H._QUERIED_CAPACITY.items()}
    saved_l2 = dict(H._QUERIED_L2)
    try:
        H.register_capacity(fam, sm, 2, 70)
        H.register_l2(fam, sm, 1.0e8)
        H.rank(300, 4096, 4096, 1, "k", 2, sm, fam)
        size = H.rank.cache_info().currsize
        assert size >= 1
        H.register_capacity(fam, sm, 2, 70)
        H.register_l2(fam, sm, 1.0e8)
        assert H.rank.cache_info().currsize == size
        H.register_capacity(fam, sm, 2, 71)
        assert H.rank.cache_info().currsize == 0
    finally:
        H._QUERIED_CAPACITY.clear()
        H._QUERIED_CAPACITY.update(saved_cap)
        H._QUERIED_L2.clear()
        H._QUERIED_L2.update(saved_l2)
        H.rank.cache_clear()


@needs_device
def test_failed_occupancy_query_is_retried_a_bounded_number_of_times(monkeypatch):
    """A device whose occupancy query fails gets the part's nominal capacity; the query is retried on later calls a bounded
    number of times, then the nominal value is kept, so a process on such a device does not re-query and re-rank on every
    call."""
    from flashinfer.gemm.batch_invariant import api
    from flashinfer.gemm.batch_invariant import heuristic as H

    dev = torch.device("cuda", torch.cuda.current_device())
    index = dev.index
    sm, fam = api.part_of(dev)
    saved = (
        dict(api._CO_RESIDENT),
        set(api._REGISTERED_DEVICES),
        dict(api._QUERY_FAILURES),
        {k: dict(v) for k, v in H._QUERIED_CAPACITY.items()},
    )
    calls: list = []

    def failing(device_index, cluster_size):
        calls.append(cluster_size)
        raise RuntimeError("no occupancy query on this device")

    monkeypatch.setattr(api, "_query_co_resident_clusters", failing)
    try:
        for k in [k for k in api._CO_RESIDENT if k[0] == index]:
            del api._CO_RESIDENT[k]
        api._REGISTERED_DEVICES.discard(index)
        api._QUERY_FAILURES.clear()
        cfg = Knobs((128, 128), (2, 1), False, True, "m", 1)
        first = api.persistent_clusters(dev, cfg)
        assert first >= 1
        for _ in range(api._QUERY_ATTEMPTS + 2):
            assert api.persistent_clusters(dev, cfg) == first
        sizes = {size for size in calls}
        assert len(calls) <= api._QUERY_ATTEMPTS * len(sizes)
        settled = len(calls)
        for _ in range(10):
            assert api.persistent_clusters(dev, cfg) == first
        assert len(calls) == settled  # no further queries once the retries are spent
        assert index in api._REGISTERED_DEVICES
    finally:
        api._CO_RESIDENT.clear()
        api._CO_RESIDENT.update(saved[0])
        api._REGISTERED_DEVICES.clear()
        api._REGISTERED_DEVICES.update(saved[1])
        api._QUERY_FAILURES.clear()
        api._QUERY_FAILURES.update(saved[2])
        H._QUERIED_CAPACITY.clear()
        H._QUERIED_CAPACITY.update(saved[3])
        H.rank.cache_clear()


@needs_device
@pytest.mark.parametrize("M,N,K", [(16, 4096, 4096), (37, 128, 8192)])
def test_public_output_buffer_and_graph_replay(M, N, K):
    """A provided output gives the exact product after capture and replay."""
    a, b = int_bf16(M, K, 181), int_bf16(K, N, 182)
    bias = frac_bias(N, 183)
    for kw in ({}, {"bias": bias}):
        ref = bits(exact_biased_reference(a, b, kw.get("bias", torch.zeros_like(bias))))
        assert torch.equal(bits(bi_mm(a, b, **kw)), ref)
        out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
        assert bi_mm(a, b, out=out, **kw) is out and torch.equal(bits(out), ref)
        chunks = chunks_for(N, K)
        if kw and chunks > 1:
            storage = torch.full(
                (chunks * M * N + 8,),
                float("nan"),
                dtype=torch.float32,
                device=a.device,
            )
            scratch = storage[4:-4]
            out.fill_(float("nan"))
            with patch.object(
                gemm_base,
                "_execute_cudnn_batch_invariant",
                wraps=gemm_base._execute_cudnn_batch_invariant,
            ) as execute:
                assert (
                    mm_bf16_batch_invariant(a, b, out=out, bias=bias, workspace=scratch)
                    is out
                )
            used = (
                execute.call_args.args[0].get_workspace_size() // scratch.element_size()
            )
            assert torch.equal(bits(out), ref)
            assert (
                torch.isnan(storage[:4]).all()
                and torch.isnan(storage[4 + used :]).all()
            )
            if used:
                assert torch.isfinite(scratch[:used]).any()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            bi_mm(a, b, out=out, **kw)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                bi_mm(a, b, out=out, **kw)
        torch.cuda.current_stream().wait_stream(s)
        out.zero_()
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(bits(out), ref)
    buf = torch.empty(M, K, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(
        bi.BatchInvariantUnsupportedError, match="A overlapping the output"
    ):
        bi_mm(buf, b, out=buf.view(-1)[: M * N].view(M, N))
    with pytest.raises(
        bi.BatchInvariantUnsupportedError, match="B overlapping the output"
    ):
        bi_mm(a, b, out=b[:M])
    out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(
        bi.BatchInvariantUnsupportedError, match="bias overlapping the output"
    ):
        bi_mm(a, b, bias=out[0], out=out)


@needs_device
def test_unsupported_inputs_raise_and_nothing_falls_back():
    a, b = rnd_bf16(16, 4096, 101), rnd_bf16(4096, 4096, 102)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="K > 0 and N > 0"):
        bi_mm(a, torch.empty(4096, 0, dtype=torch.bfloat16, device="cuda"))
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="K > 0 and N > 0"):
        bi_mm(a[:, :0], b[:0])
    off = torch.empty(4096 * 4096 + 8, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="16-byte-aligned B"):
        bi_mm(a, off[1 : 1 + 4096 * 4096].view(4096, 4096))
    with pytest.raises(
        bi.BatchInvariantUnsupportedError, match="16-byte-aligned output"
    ):
        bi_mm(a, b, out=off[1 : 1 + 16 * 4096].view(16, 4096))
    with pytest.raises(bi.BatchInvariantUnsupportedError, match='backend="auto"'):
        flashinfer.mm_bf16(a, b, backend="cudnn", batch_invariant=True)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="bf16 output"):
        bi_mm(a, b, out_dtype=torch.float32)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="multiples of 8"):
        bi_mm(rnd_bf16(16, 100, 103), rnd_bf16(100, 64, 104))
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="row-major"):
        mm_bf16_batch_invariant(
            a,
            torch.empty(4096, 2 * 4096, dtype=torch.bfloat16, device="cuda").t()[
                ::2, :
            ],
        )
    out = bi_mm(a[:0], b)
    assert out.shape == (0, 4096) and out.dtype == torch.bfloat16
    # the public API hands every batch-invariant call to the class without running the backend checks, so a
    # wrong-dtype `out` or bias raises the class's typed error there too (a ValueError, as before)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="out"):
        bi_mm(a, b, out=torch.empty(16, 4096, dtype=torch.float16, device="cuda"))
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="out"):
        mm_bf16_batch_invariant(
            a, b, torch.empty(16, 4096, dtype=torch.float16, device="cuda")
        )
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="bf16 bias"):
        bi_mm(a, b, bias=torch.zeros(4096, dtype=torch.float32, device="cuda"))
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="bf16 bias"):
        mm_bf16_batch_invariant(
            a, b, bias=torch.zeros(4096, dtype=torch.float32, device="cuda")
        )


@needs_device
def test_cudnn_plan_options_and_output_aliasing():
    a, b = adv_bf16(16, 4096, 151).cuda(), adv_bf16(4096, 4096, 152).cuda()
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="cuDNN"):
        mm_bf16_batch_invariant(
            a, b, config=Knobs((128, 64), (1, 1), False, True, "m", 1)
        )
    # the output may not alias an operand (the bits would depend on which CTA runs first)
    buf = torch.empty(16, 4096, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(
        bi.BatchInvariantUnsupportedError, match="A overlapping the output"
    ):
        mm_bf16_batch_invariant(buf, b, out=buf)
    with pytest.raises(
        bi.BatchInvariantUnsupportedError, match="B overlapping the output"
    ):
        mm_bf16_batch_invariant(a, b, out=b[:16])
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="chunks"):
        mm_bf16_batch_invariant(a, b, chunks=0)


# ----------------------------------------------------------------------------------------------- the biased chain


@needs_device
@pytest.mark.parametrize(
    "M,N,K",
    [
        (1000, 4096, 4096),
        (37, 128, 8192),
        (300, 64, 16384),
        (
            37,
            72,
            8192,
        ),  # a partial last tile: the bias read one element at a time, through the split class
        (300, 4008, 4096),  # ... and through the unsplit class
    ],
)
def test_bias_matches_the_exact_reference(M, N, K):
    """The biased chain against an exact oracle: on integer-valued operands every fp32 intermediate of the fold is exact
    and the fp32 sum with a bias of eighths is exact, so the only rounding is the class's single ``cvt.rn`` after the
    bias add; the bits must equal ``cvt_bf16(acc + bias)`` computed exactly (a double rounding -- the bias added
    after the conversion, or a bf16 bias add -- differs on these operands). Both implementations of the split class
    (workspace split, in-CTA chunk fold) and both weight layouts; the unsplit class at N = 4096."""
    a, b, bias = int_bf16(M, K, 121), int_bf16(K, N, 122), frac_bias(N, 123)
    expected = bits(exact_biased_reference(a, b, bias))
    assert torch.equal(bits(bi_mm(a, b, bias=bias)), expected)
    assert torch.equal(bits(bi_mm(a, k_major(b), bias=bias)), expected)
    # the biased and the unbiased chain are different specifications (the fractional bias moves the rounding)
    assert not torch.equal(bits(bi_mm(a, b)), expected)
    # a rounding that only the specified placement gives: rows of zeros return the bias itself, bit for bit
    z = torch.zeros(3, K, dtype=torch.bfloat16, device="cuda")
    assert torch.equal(bits(bi_mm(z, b, bias=bias)), bits(bias).expand(3, N))
    # the accuracy yardstick on general operands: within torch.addmm's (cuBLAS, fp32 accumulate, fused bias)
    a, b = rnd_bf16(M, K, 124), rnd_bf16(K, N, 125)
    bias = rnd_bf16(1, N, 126)[0]
    ref = a.double() @ b.double() + bias.double()
    scale = a.double().abs() @ b.double().abs() + bias.double().abs()
    ours = ((bi_mm(a, b, bias=bias).double() - ref).abs() / (scale + 1e-30)).max()
    theirs = ((torch.addmm(bias, a, b).double() - ref).abs() / (scale + 1e-30)).max()
    assert ours.item() <= 1.1 * theirs.item() + 1e-6, (ours.item(), theirs.item())


@needs_device
@pytest.mark.parametrize("N,K", [(4096, 4096), (128, 8192)])
def test_bias_batch_invariance(N, K):
    """R2 for the biased chain, both classes: a row's bits do not depend on the batch size, its position, its
    neighbours (a mixed batch), the order (permutation) or the selection (sub-batch)."""
    a, b, bias = adv_bf16(2049, K, 141), adv_bf16(K, N, 142), adv_bf16(1, N, 143)[0]
    full = bits(bi_mm(a, b, bias=bias))
    for M in (1, 16, 37, 128, 1000, 2049):
        assert torch.equal(bits(bi_mm(a[:M].clone(), b, bias=bias)), full[:M]), f"M={M}"
    x = torch.cat([adv_bf16(5, K, 144), a[100:101], adv_bf16(7, K, 145)])
    assert torch.equal(bits(bi_mm(x, b, bias=bias))[5], full[100])
    perm = torch.randperm(2049, generator=torch.Generator().manual_seed(1)).cuda()
    assert torch.equal(bits(bi_mm(a[perm].contiguous(), b, bias=bias)), full[perm])
    idx = torch.randperm(2049, generator=torch.Generator().manual_seed(2))[:37].cuda()
    assert torch.equal(bits(bi_mm(a[idx].contiguous(), b, bias=bias)), full[idx])


@needs_device
def test_unsupported_bias_inputs_raise():
    a, b = rnd_bf16(16, 4096, 151), rnd_bf16(4096, 4096, 152)
    N = 4096
    bias = rnd_bf16(1, N, 153)[0]
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="bf16 bias"):
        mm_bf16_batch_invariant(a, b, bias=bias.float())
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="bias of type list"):
        mm_bf16_batch_invariant(a, b, bias=[0.0] * N)
    # the chain reads the bias while it writes the output: they may not overlap, on the direct store (the unsplit
    # class, the in-CTA fold) and on the workspace split, whose fixup reads the bias
    out = torch.empty(16, N, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(
        bi.BatchInvariantUnsupportedError, match="bias overlapping the output"
    ):
        bi_mm(a, b, bias=out[0], out=out)
    Ns, Ks = 128, 8192  # the split class: C = 4
    Cs = chunks_for(Ns, Ks)
    assert Cs > 1
    a_s, b_s = rnd_bf16(16, Ks, 156), rnd_bf16(Ks, Ns, 157)
    out_s = torch.empty(16, Ns, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(
        bi.BatchInvariantUnsupportedError, match="bias overlapping the output"
    ):
        bi_mm(
            a_s, b_s, bias=out_s[0], out=out_s
        )  # the dispatcher's own pick at the split shape
    with pytest.raises(ValueError):  # the public API's own dtype contract comes first
        bi_mm(a, b, bias=bias.float())
    with pytest.raises(bi.BatchInvariantUnsupportedError, match=r"shape \(4096,\)"):
        bi_mm(a, b, bias=rnd_bf16(1, N + 8, 154)[0])
    with pytest.raises(bi.BatchInvariantUnsupportedError, match=r"shape \(4096,\)"):
        bi_mm(a, b, bias=bias.view(1, N))
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="contiguous bias"):
        bi_mm(a, b, bias=rnd_bf16(1, 2 * N, 155)[0][::2])
    off = torch.empty(N + 8, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="16-byte-aligned bias"):
        bi_mm(a, b, bias=off[1 : 1 + N])
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="bf16 output"):
        bi_mm(a, b, bias=bias, out_dtype=torch.float32)
    out = bi_mm(a[:0], b, bias=bias)
    assert out.shape == (0, N) and out.dtype == torch.bfloat16
    # an aligned slice of a larger tensor is a bias like any other
    assert torch.equal(
        bits(bi_mm(a, b, bias=off[8 : 8 + N].copy_(bias))), bits(bi_mm(a, b, bias=bias))
    )
