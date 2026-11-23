# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The batch-invariant bf16 GEMM: ``flashinfer.mm_bf16(a, b, backend="auto", batch_invariant=True)``.

There is no exact CPU oracle for ``tcgen05.mma``, so the class is checked by bit agreement across its configurations
(the hardware hypothesis of ``lean/BIKernels/GEMMConfig.lean``), by the launch-model properties R1-R2 on adversarial
operands, and by an accuracy yardstick against ``torch.matmul`` (both fp32-accumulate). The biased chain has an exact
oracle on integer-valued operands (every fp32 intermediate is exact, so the one rounding after the fp32 bias add is
checked bit for bit). Device-free tests cover the fingerprints, the chunk rule, the dispatcher and the tuner's tactic
encoding; the GPU tests need SM100 / SM103 / SM107.
"""

import pytest
import torch

import flashinfer
from flashinfer import batch_invariant as bi
from flashinfer.autotuner import AutoTuner, autotune
from flashinfer.gemm.batch_invariant import tuning
from flashinfer.gemm.batch_invariant.api import (
    MM_BF16_ARCHS,
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

_supported = torch.cuda.is_available() and bi.is_supported_device(
    torch.device("cuda"), MM_BF16_ARCHS
)
needs_device = pytest.mark.skipif(
    not _supported,
    reason="needs a CUDA device of SM100 / SM103 / SM107 with the CuTe DSL",
)


def kernel_events(prof) -> list:
    """The kernel events of a ``torch.profiler`` trace, each with its launch grid (the chrome trace's ``args.grid``)."""
    import json
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".json") as f:
        prof.export_chrome_trace(f.name)
        with open(f.name) as trace:
            return [
                e
                for e in json.load(trace)["traceEvents"]
                if e.get("cat") == "kernel" and "grid" in e.get("args", {})
            ]


_PROFILER: dict = {}


def profiler_records_kernels() -> bool:
    """Whether ``torch.profiler``'s CUDA activities record kernels on this device: kineto's CUPTI may not support the part
    under the installed toolkit (``CUPTI_ERROR_INVALID_DEVICE`` at initialisation), in which case a trace holds no kernel
    event although the kernels ran. Probed once per process with a trivial launch."""
    if "kernels" not in _PROFILER:
        from torch.profiler import ProfilerActivity, profile

        x = torch.ones(1024, device="cuda")
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            (x * 2).sum().item()
        _PROFILER["kernels"] = bool(kernel_events(prof))
    return _PROFILER["kernels"]


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


def test_kernel_handle_cache_clears_with_the_rankings():
    """The launch-handle cache has one supported clear, and forgetting the rankings (a device's registered numbers) forgets
    the handles too: a handle is sized with the grid the rankings were computed for."""
    from flashinfer.gemm.batch_invariant import api as bi_api, heuristic

    bi_api._KERNELS[("probe",)] = object()
    bi_api.clear_kernel_cache()
    assert ("probe",) not in bi_api._KERNELS
    bi_api._KERNELS[("probe",)] = object()
    heuristic.rank.cache_clear()
    assert ("probe",) not in bi_api._KERNELS


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


def test_autotuner_lookup_is_skipped_exactly_when_it_cannot_hit(monkeypatch):
    """``AutoTuner._has_tune_data(op)`` is the public path's precondition for the autotuner's lookup: False only when every
    source the lookup consults is empty for the op (no tuning mode, no replay diagnostics, no managed store, no loaded or
    bundled configs, no in-memory winner of the op); it follows every publication and ``clear_cache``."""
    from flashinfer.autotuner.autotuner import ProfilingCacheKey

    monkeypatch.delenv("FLASHINFER_AUTOTUNER_LOAD_FROM_FILE", raising=False)
    tuner = AutoTuner()  # a fresh instance: not the process singleton
    assert not tuner._has_tune_data(tuning.CUSTOM_OP)
    tuner.is_tuning_mode = True
    assert tuner._has_tune_data(tuning.CUSTOM_OP)
    tuner.is_tuning_mode = False
    assert not tuner._has_tune_data(tuning.CUSTOM_OP)  # the memoised negative
    key = ProfilingCacheKey(tuning.CUSTOM_OP_MM_FP4, "R", 1, ((1, 2),), ())
    tuner._cache_profile_result(key, None, TUNING_CONFIG_OF_TEST, (1, 2), {})
    assert tuner._has_tune_data(tuning.CUSTOM_OP_MM_FP4)
    assert not tuner._has_tune_data(tuning.CUSTOM_OP)  # another op's winner is not ours
    tuner._file_configs["x"] = (
        "R",
        -1,
    )  # a loaded cache: any op may hit (the file may name it)
    assert tuner._has_tune_data(tuning.CUSTOM_OP)
    tuner.clear_cache()
    assert not tuner._has_tune_data(tuning.CUSTOM_OP_MM_FP4)
    monkeypatch.setenv("FLASHINFER_AUTOTUNER_LOAD_FROM_FILE", "1")
    assert tuner._has_tune_data(tuning.CUSTOM_OP)


TUNING_CONFIG_OF_TEST = tuning.TUNING_CONFIG


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


def test_tactic_encoding_round_trips():
    for cfg in enumerate_space(16, 128, 8192, chunks=4)[::97]:
        t = tuning.knobs_to_tactic(cfg)
        assert (
            tuning.knobs_from_tactic(t) == cfg
            and tuning.knobs_from_tactic(list(t)) == cfg
        )


def test_scheduler_decode_mirror_visits_every_tile_once():
    """The host mirror of the kernel's tile decode (``scheduler.tile_order``): a permutation of the cluster tiles for
    every swizzle, raster order, batch extent and odd tile count; ``s = 1`` is the plain raster order; the full swizzle
    bands keep the static scheduler's order (bands ascending, the slow index ascending inside a band, the band's fast
    tiles innermost) and the remainder band holds exactly the tiles the static scheduler padded with phantoms."""
    from flashinfer.gemm.batch_invariant.blocks.scheduler import tile_order

    for mt in (1, 2, 3, 7, 16):
        for nt in (1, 3, 5, 8, 13):
            for L in (1, 2):
                for raster_m in (True, False):
                    plain = [
                        (m, n, l)
                        for l in range(L)
                        for f in range(nt if raster_m else mt)
                        for sl in range(mt if raster_m else nt)
                        for (m, n) in [((sl, f) if raster_m else (f, sl))]
                    ]
                    assert tile_order(mt, nt, L, 1, raster_m) == plain
                    for s in (2, 4, 8):
                        order = tile_order(mt, nt, L, s, raster_m)
                        assert sorted(order) == sorted(plain), (mt, nt, L, s, raster_m)
                        S, F = (mt, nt) if raster_m else (nt, mt)
                        static = [
                            ((sl, f, l) if raster_m else (f, sl, l))
                            for l in range(L)
                            for band in range(-(-F // s))
                            for sl in range(S)
                            for f in range(band * s, min((band + 1) * s, F))
                        ]
                        full = (F // s) * s * S
                        for l in range(L):
                            per_l = S * F
                            assert (
                                order[l * per_l : l * per_l + full]
                                == static[l * per_l : l * per_l + full]
                            )
                            assert sorted(
                                order[l * per_l + full : (l + 1) * per_l]
                            ) == sorted(static[l * per_l + full : (l + 1) * per_l])


def test_cache_key_covers_every_kernel_source():
    """The compiled-kernel cache key is the kernel sources (the blocks, the fixup, the specification and generator they
    import, the vendored tile scheduler) and not the dispatcher's host-side selection code (chunk rule, heuristic): an edit to
    the latter must keep every compiled object valid."""
    from flashinfer.batch_invariant.jit import relative_sources

    srcs = set(
        relative_sources("batch_invariant_gemm")
    )  # the GEMM cache module shared by the bf16 and fp8 classes
    assert {
        "gemm/batch_invariant/blocks/compile.py",
        "gemm/batch_invariant/blocks/epilogue.py",
        "gemm/batch_invariant/blocks/kernel.py",
        "gemm/batch_invariant/blocks/mainloop.py",
        "gemm/batch_invariant/blocks/operands.py",
        "gemm/batch_invariant/blocks/producer.py",
        "gemm/batch_invariant/blocks/scheduler.py",
        "gemm/batch_invariant/fixup.py",
        "gemm/batch_invariant/spec.py",
        "gemm/batch_invariant/generator.py",
        "cute_dsl/attention/fmha/helpers/static_persistent_tile_scheduler.py",
    } <= srcs
    assert (
        not {
            "gemm/batch_invariant/schedule.py",
            "gemm/batch_invariant/heuristic.py",
        }
        & srcs
    )


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
    # the dispatcher's M-dependent picks do not change bits: force another M's default on the same rows
    for M in (16, 1000):
        c1 = mm_bf16_batch_invariant(
            a[:M].clone(), b, config=default_config(M, N, K, 1, "n", 204)
        )
        c2 = mm_bf16_batch_invariant(
            a[:M].clone(), b, config=default_config(2049, N, K, 1, "n", 148)
        )
        assert torch.equal(bits(c1), bits(c2)) and torch.equal(bits(c1), full[:M])


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
@pytest.mark.parametrize("M,N,K", [(1000, 4096, 4096), (37, 2048, 4000)])
def test_config_agreement(M, N, K):
    """R3: every configuration the tuner may choose (the dispatcher's top candidates on both parts) and a spread of
    hand-picked records (64- / 128- / 256-row tiles, 1- and 2-CTA MMA, clusters, both rasters, swizzles, a decode
    tile) give identical bits."""
    a, b = adv_bf16(M, K, 61), adv_bf16(K, N, 62)
    cfgs = []
    for family, sm in PARTS:
        cfgs += rank_topk(M, N, K, 1, "n", sm, 10, family)
    cfgs += [
        Knobs((64, 32), (1, 1), False, True, "m", 1),
        Knobs((128, 128), (1, 2), False, True, "n", 2),
        Knobs((128, 256), (2, 1), True, True, "m", 1),
        Knobs((256, 64), (2, 2), True, True, "n", 4),
        Knobs((128, 16), (1, 1), False, True, "m", 1),
        Knobs((64, 192), (4, 1), False, True, "m", 8),
    ]
    cfgs = [c for c in dict.fromkeys(cfgs) if can_implement(c, M, N, K, "n", 1)]
    assert len(cfgs) >= 12
    ref = bits(mm_bf16_batch_invariant(a, b, config=cfgs[0]))
    for c in cfgs[1:]:
        assert torch.equal(bits(mm_bf16_batch_invariant(a, b, config=c)), ref), (
            f"{c.describe()} differs from {cfgs[0].describe()}"
        )


@needs_device
@pytest.mark.parametrize("M,N,K", [(37, 128, 8192), (300, 64, 16384)])
def test_cta_chunk_fold_equals_workspace_split(M, N, K):
    """The split class ``kchunk[C]`` has two implementations: the workspace split (chunks on different CTAs, an fp32
    workspace, a fixup fold in chunk order) and the in-CTA chunk fold (one CTA folds every chunk of its tile through a
    TMEM running sum in chunk order). They are one term in Lean (`ctaSplit_eq_kchunk`) and one bit pattern here, on
    both weight layouts; and the split class is a different specification from the unsplit fold (expected)."""
    C = chunks_for(N, K)
    assert C > 1
    a, b = adv_bf16(M, K, 71), adv_bf16(K, N, 72)
    ws = Knobs((128, 64), (1, 1), False, True, "m", 1)
    ref = bits(mm_bf16_batch_invariant(a, b, config=ws, chunks=C))
    recs = [k for k in rank_topk(M, N, K, C, "n", 204, 10, "sm_107") if k.cta_chunks]
    recs += [
        Knobs((64, 32), (1, 4), False, True, "m", 1, 0, True),
        Knobs((128, 64), (1, 1), False, True, "n", 1, 0, True),
        Knobs((256, 128), (2, 1), True, True, "m", 2, 0, True),
    ]
    recs = [k for k in dict.fromkeys(recs) if can_implement(k, M, N, K, "n", C)]
    assert len(recs) >= 3
    for k in recs:
        assert torch.equal(
            bits(mm_bf16_batch_invariant(a, b, config=k, chunks=C)), ref
        ), k.describe()
        assert torch.equal(
            bits(mm_bf16_batch_invariant(a, k_major(b), config=k, chunks=C)), ref
        ), k.describe()
    assert torch.equal(
        bits(bi_mm(a, b)), ref
    )  # the dispatcher's own pick is a member too
    assert not torch.equal(
        bits(mm_bf16_batch_invariant(a, b, config=ws, chunks=1)), ref
    )


@needs_device
def test_workspace_split_at_large_m():
    """The split class with its partials in the workspace at M beyond the hardware's grid.y limit (the fixup kernel strides
    over rows): bit-equal to the in-CTA chunk fold and to its own sub-batch."""
    M, N, K = 70000, 128, 8192
    C = chunks_for(N, K)
    assert C == 4
    a, b = rnd_bf16(M, K, 111), rnd_bf16(K, N, 112)
    ws_rec = Knobs((128, 64), (1, 1), False, True, "m", 1)
    cta_rec = Knobs((128, 64), (1, 1), False, True, "m", 1, 0, True)
    full = bits(mm_bf16_batch_invariant(a, b, config=ws_rec, chunks=C))
    assert torch.equal(
        full, bits(mm_bf16_batch_invariant(a, b, config=cta_rec, chunks=C))
    )
    assert torch.equal(
        bits(mm_bf16_batch_invariant(a[:1000], b, config=ws_rec, chunks=C)), full[:1000]
    )
    ws = torch.empty(C * M * N + 4, dtype=torch.float32, device="cuda")
    assert torch.equal(
        bits(
            mm_bf16_batch_invariant(
                a, b, config=ws_rec, chunks=C, workspace=ws[: C * M * N]
            )
        ),
        full,
    )
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="workspace"):
        mm_bf16_batch_invariant(
            a[:16], b, config=ws_rec, chunks=C, workspace=ws[1 : 1 + C * 16 * N]
        )  # 4-byte offset
    # the workspace may not overlap what the launch reads: the operands (and the output, the bias)
    for what, kw in (
        ("A", dict(a=ws.view(torch.bfloat16)[: 16 * K].view(16, K), b=b)),
        ("B", dict(a=a[:16], b=ws.view(torch.bfloat16)[: K * N].view(N, K).t())),
    ):
        with pytest.raises(
            bi.BatchInvariantUnsupportedError, match=f"{what} overlapping the workspace"
        ):
            mm_bf16_batch_invariant(
                config=ws_rec, chunks=C, workspace=ws[: C * 16 * N], **kw
            )
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="workspace"):
        mm_bf16_batch_invariant(
            a[:16], b, config=ws_rec, chunks=C, workspace=ws[: 2 * C * 16 * N : 2]
        )  # strided


@needs_device
def test_persistent_grid_size_does_not_change_bits():
    """R2 for the SM count: a smaller persistent grid is what a smaller part, an SM carveout, MPS or a green context
    produce; it only decides which CTA computes which tile (and, for the split class, which chunk)."""
    import dataclasses

    for M, N, K in [(1000, 4096, 4096), (64, 64, 65536)]:
        a, b = adv_bf16(M, K, 81), adv_bf16(K, N, 82)
        C = chunks_for(N, K)
        base = default_config(M, N, K, C, "n", 204)
        ref = bits(mm_bf16_batch_invariant(a, b, config=base, chunks=C))
        for g in (1, 3, 17, 74):
            cfg = dataclasses.replace(base, grid_clusters=g)
            assert torch.equal(
                bits(mm_bf16_batch_invariant(a, b, config=cfg, chunks=C)), ref
            ), f"grid={g} at {(M, N, K)}"


@needs_device
@pytest.mark.parametrize(
    "N,K,layout,biased",
    [
        (4096, 4096, "n", False),
        (64, 8192, "k", False),
        (4096, 4096, "k", True),
        (128, 8192, "n", True),
    ],
)
def test_autotuning_profiles_every_candidate_and_keeps_the_bits(N, K, layout, biased):
    """Under ``autotune(True)`` the FlashInfer autotuner profiles the class's candidates (none may be dropped as
    unsupported) and the tuned process computes the same bits as the untuned one, during and after tuning; with a bias
    the biased kernels are what it profiles (the bias is an input of the runner)."""
    a = adv_bf16(300, K, 91)
    b = adv_bf16(K, N, 92) if layout == "n" else k_major(adv_bf16(K, N, 92))
    kw = {"bias": adv_bf16(1, N, 93)[0]} if biased else {}

    def bi_mm(a, b):  # noqa: F811  (the biased or unbiased call of this case)
        return flashinfer.mm_bf16(a, b, backend="auto", batch_invariant=True, **kw)

    ref = [bits(bi_mm(a[:M], b)) for M in (1, 16, 300)]
    tuner = AutoTuner.get()
    before = {
        k: len(v)
        for k, v in tuner.stats.failed_tactics.items()
        if tuning.CUSTOM_OP in k
    }
    with autotune(True):
        tuned = [bits(bi_mm(a[:M], b)) for M in (1, 16, 300)]
    after = {
        k: len(v)
        for k, v in tuner.stats.failed_tactics.items()
        if tuning.CUSTOM_OP in k
    }
    assert after == before, f"tactics failed while profiling: {after}"
    post = [bits(bi_mm(a[:M], b)) for M in (1, 16, 300)]
    for r, t, p in zip(ref, tuned, post, strict=True):
        assert torch.equal(r, t) and torch.equal(r, p)
    assert torch.equal(post[1], post[2][:16])


@needs_device
def test_public_api_takes_the_autotuner_lookup_only_where_it_may_hit(monkeypatch):
    """The public path asks the autotuner only where ``_has_tune_data`` says it may hold something; otherwise it runs the
    dispatcher's pick directly, the same configuration the lookup would have fallen back to (the same bits as the class
    entry). A process that tuned the op keeps the lookup."""
    a, b = adv_bf16(16, 4096, 171), adv_bf16(4096, 4096, 172)
    ref = bits(mm_bf16_batch_invariant(a, b))
    calls = []
    real = AutoTuner.choose_one

    def counting(self, custom_op, *args, **kw):
        calls.append(custom_op)
        return real(self, custom_op, *args, **kw)

    monkeypatch.setattr(AutoTuner, "choose_one", counting)
    monkeypatch.setattr(AutoTuner, "_has_tune_data", lambda self, op: False)
    assert torch.equal(bits(bi_mm(a, b)), ref) and calls == []
    monkeypatch.setattr(AutoTuner, "_has_tune_data", lambda self, op: True)
    assert torch.equal(bits(bi_mm(a, b)), ref) and calls == [tuning.CUSTOM_OP]
    monkeypatch.undo()
    monkeypatch.setattr(AutoTuner, "choose_one", counting)
    calls.clear()
    with autotune(True):
        assert torch.equal(bits(bi_mm(a, b)), ref)
    assert AutoTuner.get()._has_tune_data(tuning.CUSTOM_OP)  # tuned in this process
    n = len(calls)
    assert torch.equal(bits(bi_mm(a, b)), ref) and len(calls) == n + 1


@needs_device
def test_public_api_skips_backend_selection(monkeypatch):
    """batch_invariant=True never runs a backend, so the @backend_requirement wrapper must not run the auto selection
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
def test_persistent_grid_is_the_device_co_resident_cluster_count():
    """The compiled persistent grid is the launch device's co-resident cluster count for the configuration's cluster
    shape (the GPC-aware occupancy query), not SMs // cluster size: the launched grid, read from a profiler trace of a
    many-tile GEMM, is that many clusters; the count is a launch argument, not part of the object's name; the bits are
    those of any other grid. Where the profiler records no kernels (a CUPTI that does not support the device under the
    installed toolkit), the grid readback is skipped and the other checks still run."""
    import dataclasses

    import cutlass.utils as cutlass_utils
    from torch.profiler import ProfilerActivity, profile

    from flashinfer.gemm.batch_invariant.api import persistent_clusters
    from flashinfer.gemm.batch_invariant.blocks.compile import kernel_name

    dev = torch.device("cuda", torch.cuda.current_device())
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    a, b = rnd_bf16(4096, 1024, 161), rnd_bf16(1024, 4096, 162)
    ref = bits(bi_mm(a, b))
    traced = profiler_records_kernels()
    for cfg in [
        Knobs((128, 128), (1, 1), False, True, "m", 1),
        Knobs((128, 128), (2, 1), False, True, "m", 1),
        Knobs((256, 128), (2, 2), True, True, "m", 1),
    ]:
        size = cfg.cluster_m * cfg.cluster_n
        query = cutlass_utils.HardwareInfo(device_id=dev.index).get_max_active_clusters(
            size
        )
        clusters = persistent_clusters(dev, cfg)
        assert clusters == query and 0 < clusters <= sms // size, (
            cfg.describe(),
            clusters,
            query,
        )
        # the grid is a launch argument: the object's name is the same for every count
        assert kernel_name(cfg, "n", 1) == kernel_name(
            dataclasses.replace(cfg, grid_clusters=7), "n", 1
        )
        out = mm_bf16_batch_invariant(
            a, b, config=cfg
        )  # compiled before the profiled launch
        torch.cuda.synchronize()
        if traced:
            # CUPTI can drop a launch's record (#8967): retry; zeroing `out` makes the bits check prove each launch ran.
            for _ in range(3):
                out.zero_()
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    mm_bf16_batch_invariant(a, b, config=cfg, out=out)
                    torch.cuda.synchronize()
                assert torch.equal(bits(out), ref), cfg.describe()
                events = kernel_events(prof)
                if events:
                    break
            assert events, "no kernel in the trace"
            grid = max(events, key=lambda e: e["dur"])["args"]["grid"]
            launched = grid[0] * grid[1] * grid[2]
            assert launched == clusters * size, (
                cfg.describe(),
                grid,
                clusters,
            )  # 512 tiles: more than the grid
        assert torch.equal(bits(out), ref), cfg.describe()
    if not traced:
        pytest.skip(
            "torch.profiler records no CUDA kernel on this device with the installed toolkit (CUPTI unavailable): the "
            "launched grid was not read back; the occupancy query, the cache key and the bits were checked"
        )


def _recorded_builds(monkeypatch) -> list:
    """Every kernel name ``jit.build_and_load`` is asked for (the compile or load of one object), with the in-process
    caches that would short-circuit it cleared."""
    import flashinfer.batch_invariant.jit as J
    from flashinfer.gemm.batch_invariant import api as bi_api, fixup
    from flashinfer.gemm.batch_invariant.blocks import compile as bc

    names = []
    orig = J.build_and_load

    def recording(module, kernel, compile_fn):
        names.append(kernel)
        return orig(module, kernel, compile_fn)

    monkeypatch.setattr(J, "build_and_load", recording)
    bc._compiled_gemm.cache_clear()
    fixup.get_fixup.cache_clear()
    bi_api.clear_kernel_cache()
    return names


@needs_device
def test_launch_placements_share_one_compiled_object(monkeypatch):
    """Records that differ only in the raster order, the swizzle size, the persistent grid or (the split class) the chunk
    count launch ONE compiled object -- one ``jit.build_and_load`` name, one launch-handle entry, the same callable --
    with bit-identical outputs; the in-CTA fold is its own object (a static flag), the fixup its own per chunk count."""
    import dataclasses

    from flashinfer.gemm.batch_invariant import api as bi_api
    from flashinfer.gemm.batch_invariant.blocks.compile import object_record

    names = _recorded_builds(monkeypatch)
    a, b = rnd_bf16(300, 2048, 171), rnd_bf16(2048, 4096, 172)
    base = Knobs((128, 128), (2, 1), True, True, "m", 1)
    ref = bits(mm_bf16_batch_invariant(a, b, config=base))
    variants = [
        dataclasses.replace(base, raster_along=r, swizzle_size=s, grid_clusters=g)
        for r in ("m", "n")
        for s in (1, 2, 4, 8)
        for g in (0, 9)
    ]
    for cfg in variants:
        assert torch.equal(bits(mm_bf16_batch_invariant(a, b, config=cfg)), ref), (
            cfg.describe()
        )
    gemm_names = [n for n in names if n.startswith("bf16_")]
    assert (
        len(gemm_names) == 1
        and "_sw" not in gemm_names[0]
        and "_g" not in gemm_names[0]
    ), names
    # one launch handle per record (its own trailing arguments), all holding the same compiled kernel
    handles = {
        k: v
        for k, v in bi_api._KERNELS.items()
        if object_record(k[1]) == object_record(base)
    }
    assert len(handles) == len(variants), handles.keys()
    assert len({id(v[0]) for v in handles.values()}) == 1, handles.keys()
    clusters = bi_api.persistent_clusters(a.device, base)
    assert {v[1] for v in handles.values()} == {
        (g or clusters, s, 1 if r == "m" else 0)
        for r in ("m", "n")
        for s in (1, 2, 4, 8)
        for g in (0, 9)
    }
    # the split class: C = 2 and C = 8 (two weight shapes) share the partial kernel; the fixup is per chunk count
    del names[:]
    cfg = Knobs((64, 32), (1, 4), False, True, "m", 1)
    for N, K, C in [(512, 16384, 2), (256, 32768, 8)]:
        a, b = rnd_bf16(64, K, 173 + C), rnd_bf16(K, N, 174 + C)
        assert chunks_for(N, K) == C
        outs = [
            bits(mm_bf16_batch_invariant(a, b, config=k, chunks=C))
            for k in (
                cfg,
                dataclasses.replace(cfg, raster_along="n", swizzle_size=4),
                dataclasses.replace(cfg, cta_chunks=True),
            )
        ]
        assert torch.equal(outs[0], outs[1]) and torch.equal(outs[0], outs[2]), (N, K)
    gemm_names = sorted(n for n in names if n.startswith("bf16_"))
    assert len(gemm_names) == 2 and all("_CL_" in n for n in gemm_names), names
    assert sum("_cc" in n for n in gemm_names) == 1, names
    fixups = sorted(n for n in names if n.startswith("splitk_fixup"))
    assert fixups == ["splitk_fixup_C2_bf16", "splitk_fixup_C8_bf16"], names
    # one partial kernel for both chunk counts and the raster / swizzle variant (the in-CTA fold is its own object)
    kernels = {
        id(v[0])
        for k, v in bi_api._KERNELS.items()
        if object_record(k[1]) == object_record(cfg) and not k[1].cta_chunks
    }
    assert len(kernels) == 1, kernels


@needs_device
def test_bucket_tactic_sweep_compiles_one_object_per_placement_class(monkeypatch):
    """The autotuner's candidate list for one shape compiles one object per (tile, cluster, 2-CTA, in-CTA fold) class:
    its raster twins and swizzle variants are launch arguments of the same object."""
    names = _recorded_builds(monkeypatch)
    M, N, K = 512, 4096, 4096
    C = chunks_for(N, K)
    sm, fam = PARTS[0][1], PARTS[0][0]
    dev = torch.device("cuda", torch.cuda.current_device())
    from flashinfer.gemm.batch_invariant.api import part_of
    from flashinfer.gemm.batch_invariant.blocks.compile import object_record

    sm, fam = part_of(dev)
    tactics = rank_topk(M, N, K, C, "n", sm, 10, fam)
    a, b = rnd_bf16(M, K, 175), rnd_bf16(K, N, 176)
    ref = bits(mm_bf16_batch_invariant(a, b, config=tactics[0], chunks=C))
    for cfg in tactics[1:]:
        assert torch.equal(
            bits(mm_bf16_batch_invariant(a, b, config=cfg, chunks=C)), ref
        ), cfg.describe()
    gemm_names = [n for n in names if n.startswith("bf16_")]
    classes = {object_record(t) for t in tactics}
    assert len(gemm_names) == len(set(gemm_names)) == len(classes), (
        gemm_names,
        classes,
    )
    assert len(classes) < len(tactics), "the candidate list holds raster twins"


@needs_device
@pytest.mark.parametrize("static_l1", [True, False])
def test_kernel_side_decode_matches_the_host_mirror(static_l1):
    """The compiled scheduler's decode of every linear index (FastDivmod divisors, the ``s = 1`` branch, the remainder
    band, the batch slice) is the host mirror ``tile_order``, and indices beyond the tile count are invalid: a probe
    kernel with the tile counts, the swizzle and the raster flag as runtime arguments, one object for every case."""
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    from flashinfer.gemm.batch_invariant.blocks.scheduler import (
        SchedParams,
        TileScheduler,
        tile_order,
    )

    EXTRA = 5  # blocks beyond the tile count: invalid work

    class Probe:
        def __init__(self, static_l1: bool):
            self.static_l1 = static_l1

        @cute.jit
        def __call__(
            self,
            out: cute.Tensor,
            mt: cutlass.Int32,
            nt: cutlass.Int32,
            l: cutlass.Int32,
            swizzle: cutlass.Int32,
            raster_m: cutlass.Int32,
            stream: cuda.CUstream,
        ):
            if cutlass.const_expr(self.static_l1):
                params = SchedParams((mt, nt, 1), (1, 1), swizzle, raster_m)
            else:
                params = SchedParams((mt, nt, l), (1, 1), swizzle, raster_m)
            self.kernel(out, params).launch(
                grid=(1, 1, params.total + EXTRA), block=[32, 1, 1], stream=stream
            )

        @cute.kernel
        def kernel(self, out: cute.Tensor, params: SchedParams):
            tidx, _, _ = cute.arch.thread_idx()
            _, _, z = cute.arch.block_idx()
            if tidx == 0:
                sched = TileScheduler.create(
                    params, cute.arch.block_idx(), cute.arch.grid_dim()
                )
                work = sched.get_current_work()
                out[z, 0] = work.tile_idx[0]
                out[z, 1] = work.tile_idx[1]
                out[z, 2] = work.tile_idx[2]
                if work.is_valid_tile:
                    out[z, 3] = cutlass.Int32(1)

    out = torch.zeros(4096, 4, dtype=torch.int32, device="cuda")
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    probe = cute.compile(
        Probe(static_l1),
        from_dlpack(out, assumed_align=16).mark_layout_dynamic(),
        cutlass.Int32(1),
        cutlass.Int32(1),
        cutlass.Int32(1),
        cutlass.Int32(1),
        cutlass.Int32(1),
        stream,
    )
    cases = [
        (mt, nt, L, s, rm)
        for mt in (1, 3, 7, 16)
        for nt in (1, 5, 8, 13)
        for L in ((1,) if static_l1 else (2, 3))
        for s in (1, 2, 4, 8)
        for rm in (1, 0)
    ]
    for mt, nt, L, s, rm in cases:
        total = mt * nt * L
        out.zero_()
        probe(
            from_dlpack(out, assumed_align=16).mark_layout_dynamic(),
            cutlass.Int32(mt),
            cutlass.Int32(nt),
            cutlass.Int32(L),
            cutlass.Int32(s),
            cutlass.Int32(rm),
            stream,
        )
        torch.cuda.synchronize()
        got = out[: total + EXTRA].tolist()
        assert all(row[3] == 1 for row in got[:total]) and all(
            row[3] == 0 for row in got[total:]
        ), (mt, nt, L, s, rm)
        assert [tuple(row[:3]) for row in got[:total]] == tile_order(
            mt, nt, L, s, bool(rm)
        ), (mt, nt, L, s, rm)


@needs_device
@pytest.mark.parametrize("M,N,K", [(16, 4096, 4096), (37, 128, 8192)])
def test_public_path_equals_the_class_entry_and_checks_caller_buffers(M, N, K):
    """The public path validates once and launches on the prepared operands: its bits are the class entry's with its own
    output, with a caller's output and under graph replay; a caller's output that aliases an operand is still refused
    (only the entry's own fresh output skips the aliasing check)."""
    a, b = adv_bf16(M, K, 181), adv_bf16(K, N, 182)
    bias = adv_bf16(1, N, 183)[0]
    for kw in ({}, {"bias": bias}):
        ref = bits(mm_bf16_batch_invariant(a, b, **kw))
        assert torch.equal(bits(bi_mm(a, b, **kw)), ref)
        out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
        assert bi_mm(a, b, out=out, **kw) is out and torch.equal(bits(out), ref)
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
def test_chunk_count_must_be_positive():
    a, b = adv_bf16(16, 4096, 151).cuda(), adv_bf16(4096, 4096, 152).cuda()
    # a caller-forced configuration the kernel cannot implement for the shape (here the in-CTA chunk fold of the unsplit
    # class) is rejected before anything is compiled
    forced = Knobs((128, 64), (1, 1), False, True, "m", 1, 0, True)
    assert not can_implement(forced, 16, 4096, 4096, "n", 1)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="enumerated space"):
        mm_bf16_batch_invariant(a, b, config=forced, chunks=1)
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
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="chunks = 0"):
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
    C = chunks_for(N, K)
    recs = [Knobs((128, 64), (1, 1), False, True, "m", 1)]
    if C > 1:
        recs.append(Knobs((128, 64), (1, 1), False, True, "m", 1, 0, True))
    for rec in recs:
        assert torch.equal(
            bits(mm_bf16_batch_invariant(a, b, bias=bias, config=rec, chunks=C)),
            expected,
        ), rec.describe()
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
@pytest.mark.parametrize("M,N,K", [(1000, 4096, 4096), (37, 128, 8192)])
def test_bias_bits_agree_across_configurations(M, N, K):
    """R3 / R4 for the biased chain: the tuner's candidates on both parts, hand-picked records (tiles, clusters, 2-CTA,
    rasters, swizzles, a decode tile), persistent grid sizes and, for the split class, the workspace split against the
    in-CTA chunk fold, all give one bit pattern, on both weight layouts; the dispatcher's pick is a member."""
    import dataclasses

    C = chunks_for(N, K)
    a, b, bias = adv_bf16(M, K, 131), adv_bf16(K, N, 132), adv_bf16(1, N, 133)[0]
    cfgs = []
    for family, sm in PARTS:
        cfgs += rank_topk(M, N, K, C, "n", sm, 4, family)
    cfgs += [
        Knobs((64, 32), (1, 1), False, True, "m", 1),
        Knobs((128, 128), (1, 2), False, True, "n", 2),
        Knobs((256, 64), (2, 2), True, True, "n", 4),
        Knobs((128, 16), (1, 1), False, True, "m", 1),
    ]
    if C > 1:
        cfgs += [
            Knobs((128, 64), (1, 1), False, True, "m", 1),
            Knobs((64, 32), (1, 4), False, True, "m", 1, 0, True),
            Knobs((256, 128), (2, 1), True, True, "m", 2, 0, True),
        ]
    cfgs = [c for c in dict.fromkeys(cfgs) if can_implement(c, M, N, K, "n", C)]
    assert len(cfgs) >= 8
    if C > 1:
        assert any(c.cta_chunks for c in cfgs) and any(not c.cta_chunks for c in cfgs)
    ref = bits(bi_mm(a, b, bias=bias))
    for c in cfgs:
        for bb in (b, k_major(b)):
            assert torch.equal(
                bits(mm_bf16_batch_invariant(a, bb, bias=bias, config=c, chunks=C)),
                ref,
            ), c.describe()
    for g in (1, 7, 74):
        cfg = dataclasses.replace(cfgs[0], grid_clusters=g)
        assert torch.equal(
            bits(mm_bf16_batch_invariant(a, b, bias=bias, config=cfg, chunks=C)), ref
        ), f"grid={g}"


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
    ws_rec = Knobs((128, 64), (1, 1), False, True, "m", 1)
    with pytest.raises(
        bi.BatchInvariantUnsupportedError, match="bias overlapping the output"
    ):
        mm_bf16_batch_invariant(
            a_s, b_s, out=out_s, bias=out_s[0], config=ws_rec, chunks=Cs
        )
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
