# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""``flashinfer.nvfp4_quantize(a, a_global_sf, ..., backend="auto", batch_invariant=True)``: the classes ``nvfp4.quant``
(the per-tensor global scale is an input) and ``nvfp4.quant.pertoken`` (``per_token_activation=True``: the global scale
is derived from the row's amax and returned as the per-token scale).

Device-free: the pinned fingerprints, the dispatcher's membership in the enumerated space and the device check. On an
SM100 / SM103 / SM107 device with the CuTe DSL: bit equality with the exact CPU oracle of the specification
(``quantization/nvfp4_reference.py``: the vectorised model over a shape sweep and the Fractions evaluation of the
symbolic spec on a few rows) in the three scale-factor layouts and bf16 / f16 / fp32; determinism including CUDA-graph replay
and PDL; configuration and SM-count independence; the batch-invariance guarantee itself (a row's bits do not depend on
the batch it is quantized in); M = 0 and the error contract. Every assertion is bitwise, against the classes' own
specification: FlashInfer's other quantizer backends are not a reference for them.
"""

import pytest
import torch

import flashinfer
from flashinfer import batch_invariant as bi
from flashinfer.batch_invariant.quantization import layout as L
from flashinfer.batch_invariant.quantization import nvfp4_reference as R
from flashinfer.batch_invariant.quantization import nvfp4_schedule as C
from flashinfer.batch_invariant.quantization.nvfp4_api import (
    NVFP4_QUANT_ARCHS,
    nvfp4_quantize_batch_invariant,
    nvfp4_quantize_rows,
)
from tests.quantization.test_helpers.batch_invariant_quant import (
    FI_LAYOUTS,
    adversarial_inputs,
    same_bits,
)

BLOCK = 16
DT = [torch.bfloat16, torch.float16, torch.float32]
SWEEP = {
    torch.bfloat16: [
        (1, 16),
        (3, 48),
        (37, 256),
        (130, 1024),
        (8, 2048),
        (200, 4096),
        (37, 7168),
        (5, 16384),
    ],
    torch.float16: [(37, 256), (130, 1024), (5, 16384)],
    torch.float32: [(37, 256), (130, 1024), (5, 16384)],
}
SM_COUNTS = [148, 152, 204, 224]  # B200 / B300 (SM100 / SM103), Rubin (SM107) parts
#: Configuration agreement: the whole space in the 128x4 layout at two row lengths, every third member elsewhere.
CONFIG_CASES = [
    (48, "128x4", 1),
    (1024, "128x4", 1),
    (4096, "128x4", 3),
    (1024, "8x4", 3),
    (1024, "linear", 3),
]
_supported = torch.cuda.is_available() and bi.is_supported_device(
    torch.device("cuda"), NVFP4_QUANT_ARCHS
)
needs_device = pytest.mark.skipif(
    not _supported,
    reason="needs a CUDA device of SM100 / SM103 / SM107 with the CuTe DSL",
)


def _gs(x, per_token):
    """The global scale: per token the inverse base scale of the recipe, per tensor ``448 * 6 / amax`` (finite values)."""
    if per_token:
        return 1.0 / (448.0 * 6.0)
    amax = float(x.float().abs().nan_to_num(posinf=0.0, neginf=0.0).max())
    return 448.0 * 6.0 / amax if amax > 0 else 1.0


def _quantize(x, gs, layout="128x4", per_token=False, **kw):
    """The batch-invariant path through the public API; the outputs on the host."""
    outs = flashinfer.nvfp4_quantize(
        x,
        gs,
        sfLayout=FI_LAYOUTS[layout],
        per_token_activation=per_token,
        batch_invariant=True,
        **kw,
    )
    assert outs[0].dtype == torch.uint8 and outs[1].dtype == torch.uint8
    assert outs[1].dim() == 2
    return tuple(o.cpu() for o in outs)


def _logical(outs, M, K, layout):
    """Per-row views whatever the storage layout: the codes ``(M, K / 2)``, the scale bytes ``(M, K / 16)`` and, per
    token, the scales ``(M,)``."""
    sf = L.unswizzle_sf(outs[1].reshape(-1), M, K, BLOCK, layout)
    return (outs[0].reshape(M, K // 2), sf) + tuple(outs[2:])


# ----------------------------------------------------------------------------------------------- device-free


def test_fingerprints_are_pinned_and_separate_the_classes():
    """The receipts: the per-tensor class takes no dimension (its row tree is per block); the per-token class's row
    tree spans the row, so its fingerprint takes K and separates row lengths with different trees."""
    assert bi.fingerprint("nvfp4_quantize") == "nvfp4quant-f94e79cf6ea00452"
    assert (
        bi.fingerprint("nvfp4_quantize_per_token", K=4096)
        == "nvfp4quant-64f082cd689bf7f9"
    )
    assert bi.class_name("nvfp4_quantize") == "nvfp4.quant"
    assert bi.class_name("nvfp4_quantize_per_token") == "nvfp4.quant.pertoken"
    per_token = {
        bi.fingerprint("nvfp4_quantize_per_token", K=K) for K in (16, 48, 1024, 4096)
    }
    assert len(per_token) == 4 and bi.fingerprint("nvfp4_quantize") not in per_token
    with pytest.raises(TypeError):
        bi.fingerprint("nvfp4_quantize", K=4096)
    with pytest.raises(TypeError):
        bi.fingerprint("nvfp4_quantize_per_token")


@pytest.mark.parametrize("per_token", [False, True])
def test_dispatcher_picks_are_members_of_the_space(per_token):
    """R3 membership: every configuration the closed-form dispatcher returns, for every layout, element width and the
    SM counts of the SM100 / SM103 / SM107 parts, is a member of the enumerated space the agreement test covers; the
    tile placement serves the per-tensor class in the 128x4 layout only."""
    for K in (16, 48, 256, 1024, 4096, 16384):
        for eb in (2, 4):
            space = set(C.all_valid_configs(K, eb, per_token))
            assert space
            for layout in L.LAYOUTS:
                for sm in SM_COUNTS + [None]:
                    for M in (1, 2, 16, 37, 128, 1024, 16384):
                        cfg = C.default_config(K, M, eb, per_token, sm, layout)
                        assert cfg in space, (K, eb, layout, sm, M, cfg.describe())
                        assert cfg.mode != "tile" or (
                            layout == "128x4" and not per_token
                        )


def test_unsupported_device_raises_before_any_kernel():
    x = torch.zeros(4, 64, dtype=torch.bfloat16)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="nvfp4.quant"):
        nvfp4_quantize_batch_invariant(x, 1.0)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="nvfp4.quant.pertoken"):
        nvfp4_quantize_batch_invariant(x, 1.0, per_token_activation=True)


# ----------------------------------------------------------------------------------------------- on the device


@needs_device
@pytest.mark.parametrize("per_token", [False, True])
@pytest.mark.parametrize("layout", L.LAYOUTS)
@pytest.mark.parametrize("dtype", DT)
def test_matches_exact_reference(dtype, layout, per_token):
    """Codes, scale bytes (compared in logical row order, whatever the storage layout) and per-token scales equal the
    exact CPU oracle over the shape sweep, with the recipe's corners in the inputs; the scale view has the documented
    padded shape with zero padding; a few rows equal the Fractions evaluation of the symbolic spec."""
    for M, K in SWEEP[dtype]:
        x = adversarial_inputs(M, K, dtype, seed=M + K)
        gs = _gs(x, per_token)
        outs = _quantize(x, gs, layout, per_token)
        assert outs[0].shape == (M, K // 2)
        assert outs[1].shape == (
            L.padded_sf_rows(M, layout),
            L.padded_sf_cols(K, BLOCK, layout),
        )
        got = _logical(outs, M, K, layout)
        ref_codes, ref_sf, ref_pts = R.quantize_reference(x, gs, per_token=per_token)
        assert torch.equal(got[0], ref_codes), ("codes", M, K)
        assert torch.equal(got[1], R.unswizzle_sf_128x4(ref_sf, M, K)), ("scales", M, K)
        assert int(outs[1].sum()) == int(got[1].sum()), "padding bytes are not zero"
        if per_token:
            assert torch.equal(got[2].view(torch.int32), ref_pts.view(torch.int32)), (
                "per-token scales",
                M,
                K,
            )
    rows = [0, 1, 4, 36]
    x = adversarial_inputs(37, 256, dtype, seed=7)
    gs = _gs(x, per_token)
    got = _logical(_quantize(x, gs, layout, per_token), 37, 256, layout)
    ex_codes, ex_sf, ex_pts = R.exact_rows(x, gs, rows, per_token=per_token)
    assert torch.equal(got[0][rows], ex_codes) and torch.equal(got[1][rows], ex_sf)
    if per_token:
        assert torch.equal(got[2][rows].view(torch.int32), ex_pts.view(torch.int32))


@needs_device
@pytest.mark.parametrize("per_token", [False, True])
def test_determinism_graph_replay_and_pdl(per_token):
    x = adversarial_inputs(37, 1024, torch.bfloat16)
    gs = torch.full((1,), _gs(x, per_token), dtype=torch.float32, device="cuda")
    ref = _quantize(x, gs, "128x4", per_token)
    for _ in range(5):
        assert same_bits(_quantize(x, gs, "128x4", per_token), ref)
    for pdl in (False, True):
        assert same_bits(_quantize(x, gs, "128x4", per_token, enable_pdl=pdl), ref)
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        outs = flashinfer.nvfp4_quantize(
            x, gs, per_token_activation=per_token, batch_invariant=True
        )
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            outs = flashinfer.nvfp4_quantize(
                x, gs, per_token_activation=per_token, batch_invariant=True
            )
    for _ in range(2):
        for o in outs:
            o.zero_()
        g.replay()
        torch.cuda.synchronize()
        assert same_bits(tuple(o.cpu() for o in outs), ref)


@needs_device
@pytest.mark.parametrize("per_token", [False, True])
@pytest.mark.parametrize("layout", L.LAYOUTS)
def test_batch_permutation_subbatch_invariance(layout, per_token):
    """A row's bits do not depend on the batch it is quantized in. Against a batch of 300 rows: every row alone at the
    padding boundaries, rows embedded among unrelated rows and in a much larger batch, random orders and the reverse,
    sub-batches of every size class (random, contiguous, strided, with repeated rows). The sizes span the dispatcher's
    regimes, so different sub-batches run under different configurations."""
    K, M = 1024, 300
    g = torch.Generator().manual_seed(K)
    xb = adversarial_inputs(M, K, torch.bfloat16, seed=1)
    gs = _gs(xb, per_token)

    def rows(x):
        return _logical(
            _quantize(x.contiguous(), gs, layout, per_token), x.shape[0], K, layout
        )

    def check(got, idx):
        idx = torch.as_tensor(idx)
        assert same_bits(got, tuple(t[idx] for t in full))

    full = rows(xb)
    for i in (0, 1, 7, 31, 127, 128, 129, 255, 256, 299):
        check(rows(xb[i : i + 1]), [i])
    other = adversarial_inputs(23, K, torch.bfloat16, seed=5, special=False)
    for pos, i in ((0, 17), (4, 130), (22, 299)):
        xo = other.clone()
        xo[pos] = xb[i]
        check(tuple(t[pos : pos + 1] for t in rows(xo)), [i])
    big = adversarial_inputs(2048, K, torch.bfloat16, seed=9, special=False)
    where = torch.randperm(2048, generator=g)[:M]
    big[where.cuda()] = xb
    check(tuple(t[where] for t in rows(big)), torch.arange(M))
    for seed in range(3):
        perm = torch.randperm(M, generator=torch.Generator().manual_seed(seed))
        check(rows(xb[perm.cuda()]), perm)
    rev = torch.arange(M - 1, -1, -1)
    check(rows(xb[rev.cuda()]), rev)
    for n in (1, 2, 3, 5, 8, 16, 31, 64, 100, 127, 128, 129, 200, 257, 299):
        idx = torch.randperm(M, generator=g)[:n]
        check(rows(xb[idx.cuda()]), idx)
    for lo, hi in ((0, 16), (0, 129), (37, 101), (128, 256), (171, 300), (299, 300)):
        check(rows(xb[lo:hi]), torch.arange(lo, hi))
    for step, start in ((2, 0), (3, 1), (7, 5), (128, 3)):
        idx = torch.arange(start, M, step)
        check(rows(xb[idx.cuda()]), idx)
    idx = torch.tensor([17, 17, 130, 17, 299, 130, 0, 0])
    check(rows(xb[idx.cuda()]), idx)


@needs_device
@pytest.mark.parametrize("per_token", [False, True])
@pytest.mark.parametrize("K,layout,stride", CONFIG_CASES)
def test_config_agreement(K, layout, stride, per_token):
    """R3: the members of the enumerated configuration space (lanes per block, blocks per thread, rows per CTA, the
    register / shared staging, the tile placements of the 128x4 layout) give the same bits; fp32 inputs over their
    space at one row length."""
    for dtype, eb in ((torch.bfloat16, 2), (torch.float32, 4)):
        if dtype == torch.float32 and (K != 1024 or layout != "128x4"):
            continue
        x = adversarial_inputs(37, K, dtype, seed=K)
        gs = _gs(x, per_token)
        ref = tuple(
            o.cpu() for o in nvfp4_quantize_rows(x, gs, layout, per_token=per_token)
        )
        cfgs = [
            c
            for c in C.all_valid_configs(K, eb, per_token)
            if c.mode != "tile" or layout == "128x4"
        ]
        picked = C.default_config(K, 37, eb, per_token, None, layout)
        assert picked in cfgs
        for cfg in cfgs[::stride] + [picked]:
            got = nvfp4_quantize_rows(x, gs, layout, per_token=per_token, config=cfg)
            assert same_bits(tuple(o.cpu() for o in got), ref), (layout, cfg.describe())
        assert len(cfgs) >= 4


@needs_device
def test_sm_count_does_not_change_bits():
    """The dispatcher takes the device's SM count; its picks for the SM100 / SM103 / SM107 parts may differ, the bits
    never."""
    x = adversarial_inputs(2048, 4096, torch.bfloat16, seed=9)
    for per_token in (False, True):
        gs = _gs(x, per_token)
        ref = tuple(
            o.cpu()
            for o in nvfp4_quantize_batch_invariant(
                x, gs, per_token_activation=per_token
            )
        )
        picks = set()
        for sm in SM_COUNTS:
            got = nvfp4_quantize_batch_invariant(
                x, gs, per_token_activation=per_token, sm_count=sm
            )
            assert same_bits(tuple(o.cpu() for o in got), ref), sm
            picks.add(
                C.default_config(4096, 2048, 2, per_token, sm, "128x4").describe()
            )
        print(
            f"per_token={per_token}: dispatcher picks over {SM_COUNTS}: {sorted(picks)}"
        )


@needs_device
def test_widenings_and_layouts_carry_the_same_bits():
    """One class per variant: fp32 inputs give the bits of the 16-bit inputs they exactly widen, and the three layouts
    carry the same bytes (the layout is a placement of the store)."""
    for per_token in (False, True):
        for dtype in (torch.bfloat16, torch.float16):
            x = adversarial_inputs(130, 2048, dtype, seed=3)
            gs = _gs(x, per_token)
            for layout in L.LAYOUTS:
                assert same_bits(
                    _quantize(x.float(), gs, layout, per_token),
                    _quantize(x, gs, layout, per_token),
                ), (dtype, layout)
        for M, K in ((300, 1024), (37, 96), (1000, 4096)):
            x = adversarial_inputs(M, K, torch.bfloat16, seed=M)
            gs = _gs(x, per_token)
            ref = _logical(_quantize(x, gs, "128x4", per_token), M, K, "128x4")
            for layout in ("8x4", "linear"):
                assert same_bits(
                    _logical(_quantize(x, gs, layout, per_token), M, K, layout), ref
                ), layout


@needs_device
def test_zero_rows_and_unsupported_inputs():
    """M = 0 returns the documented shapes without a launch; a 3-D input keeps its leading shape; a view with an
    under-aligned row stride gives the bits of its contiguous copy; inputs outside the class raise
    BatchInvariantUnsupportedError through the public API (never an assertion, never a fallback)."""
    for layout in L.LAYOUTS:
        x0 = torch.empty(0, 64, dtype=torch.bfloat16, device="cuda")
        fp4, sf, pts = flashinfer.nvfp4_quantize(
            x0,
            1.0 / 2688,
            sfLayout=FI_LAYOUTS[layout],
            per_token_activation=True,
            batch_invariant=True,
        )
        assert fp4.shape == (0, 32) and sf.shape == (0, 4) and pts.shape == (0,)
        fp4, sf = flashinfer.nvfp4_quantize(
            torch.empty(0, 3, 64, dtype=torch.bfloat16, device="cuda"),
            1.0,
            sfLayout=FI_LAYOUTS[layout],
            batch_invariant=True,
        )
        assert fp4.shape == (0, 3, 32) and sf.shape == (0, 4)
    x3 = adversarial_inputs(36, 48, torch.bfloat16, seed=3).view(3, 12, 48)
    gs = _gs(x3, False)
    c = _quantize(x3, gs)
    d = _quantize(x3.view(36, 48), gs)
    assert (
        c[0].shape == (3, 12, 24)
        and torch.equal(c[0].view(36, 24), d[0])
        and torch.equal(c[1], d[1])
    )
    wide = adversarial_inputs(8, 4100, torch.bfloat16, seed=7)
    view = wide[:, :4096]
    assert view.stride(0) * 2 % 16 != 0
    assert same_bits(_quantize(view, gs), _quantize(view.contiguous(), gs))
    x = adversarial_inputs(4, 1024, torch.bfloat16)
    E = bi.BatchInvariantUnsupportedError
    with pytest.raises(ValueError, match="requires backend='auto'"):
        flashinfer.nvfp4_quantize(x, gs, backend="cuda", batch_invariant=True)
    with pytest.raises(E, match="multiple of 16"):
        flashinfer.nvfp4_quantize(x[:, :1016], gs, batch_invariant=True)
    with pytest.raises(E, match="rows of at most"):
        flashinfer.nvfp4_quantize(
            torch.zeros(2, 65552, dtype=torch.bfloat16, device="cuda"),
            gs,
            batch_invariant=True,
        )
    with pytest.raises(E, match="dtype"):
        flashinfer.nvfp4_quantize(x.to(torch.float8_e4m3fn), gs, batch_invariant=True)
    with pytest.raises(E, match="CUDA device"):
        flashinfer.nvfp4_quantize(x.cpu(), gs, batch_invariant=True)
    with pytest.raises(E, match="sf_vec_size"):
        flashinfer.nvfp4_quantize(x, gs, sf_vec_size=32, batch_invariant=True)
    with pytest.raises(E, match="do_shuffle"):
        flashinfer.nvfp4_quantize(
            x, gs, sfLayout=FI_LAYOUTS["8x4"], do_shuffle=True, batch_invariant=True
        )
    with pytest.raises(E, match="multiple of 32 rows"):
        flashinfer.nvfp4_quantize(
            adversarial_inputs(300, 1024, torch.bfloat16),
            gs,
            do_shuffle=True,
            batch_invariant=True,
        )
    with pytest.raises(E, match="expanded_idx_to_permuted_idx"):
        flashinfer.nvfp4_quantize(
            x,
            gs,
            per_token_activation=True,
            batch_invariant=True,
            expanded_idx_to_permuted_idx=torch.arange(
                4, device="cuda", dtype=torch.int32
            ),
        )
    with pytest.raises(E, match="4over6"):
        nvfp4_quantize_batch_invariant(x, gs, nvfp4_4over6=object())
    with pytest.raises(E, match="out_scale"):
        nvfp4_quantize_batch_invariant(x, gs, out_scale=torch.ones(4, device="cuda"))


@needs_device
def test_fp4_quantize_entry_point_runs_the_class():
    """``flashinfer.fp4_quantize(..., batch_invariant=True)``, the entry point behind ``nvfp4_quantize``, runs the per-tensor
    class with its own argument forms: the layout flags, ``global_scale`` None as the scale 1.0, ``is_global_scale_inversed``
    as one correctly rounded division, a column-major input quantized as the transpose of its row-major view with the
    outputs transposed back; the same bytes and views as ``nvfp4_quantize``; the recipes it does not have raise."""
    from flashinfer.quantization.nvfp4_quantization_utils import NVFP44Over6Config

    x = adversarial_inputs(128, 1024, torch.bfloat16, seed=5)
    gs_t = torch.full((1,), _gs(x, False), dtype=torch.float32, device="cuda")

    def host(outs):
        return tuple(t.cpu() for t in outs)

    for layout, flags in (
        ("128x4", dict(is_sf_swizzled_layout=True)),
        ("8x4", dict(is_sf_swizzled_layout=True, is_sf_8x4_layout=True)),
        ("linear", dict(is_sf_swizzled_layout=False)),
    ):
        got = flashinfer.fp4_quantize(x, gs_t, 16, batch_invariant=True, **flags)
        ref = flashinfer.nvfp4_quantize(
            x, gs_t, sfLayout=FI_LAYOUTS[layout], batch_invariant=True
        )
        assert got[0].shape == ref[0].shape and got[1].shape == ref[1].shape
        assert same_bits(host(got), host(ref)), layout
    one = flashinfer.fp4_quantize(x, None, 16, batch_invariant=True)
    assert same_bits(
        host(one), host(flashinfer.nvfp4_quantize(x, 1.0, batch_invariant=True))
    )
    inv_t = 1.0 / gs_t
    inv = flashinfer.fp4_quantize(
        x, inv_t, 16, is_global_scale_inversed=True, batch_invariant=True
    )
    assert same_bits(
        host(inv), host(flashinfer.nvfp4_quantize(x, 1.0 / inv_t, batch_invariant=True))
    )
    y = x.t().contiguous()  # row-major (1024, 128)
    xc = y.t()  # x's values as a column-major (128, 1024) view
    assert xc.stride(-2) == 1 and torch.equal(xc, x)
    cm = flashinfer.fp4_quantize(xc, gs_t, 16, batch_invariant=True)
    rm = flashinfer.fp4_quantize(y, gs_t, 16, batch_invariant=True)
    assert torch.equal(cm[0], rm[0].t()) and torch.equal(cm[1], rm[1].t())
    E = bi.BatchInvariantUnsupportedError
    with pytest.raises(ValueError, match="requires backend='auto'"):
        flashinfer.fp4_quantize(x, gs_t, 16, backend="cuda", batch_invariant=True)
    with pytest.raises(E, match="sf_vec_size = 32 with sf_use_ue8m0 = False"):
        flashinfer.fp4_quantize(x, gs_t, 32, sf_use_ue8m0=False, batch_invariant=True)
    with pytest.raises(E, match="sf_vec_size = 16 with sf_use_ue8m0 = True"):
        flashinfer.fp4_quantize(x, None, 16, sf_use_ue8m0=True, batch_invariant=True)
    with pytest.raises(E, match="dtype"):
        flashinfer.fp4_quantize(
            x.to(torch.float8_e4m3fn), gs_t, 16, batch_invariant=True
        )
    with pytest.raises(E, match="4over6"):
        flashinfer.fp4_quantize(
            x,
            gs_t,
            16,
            nvfp4_4over6=NVFP44Over6Config(
                e4m3_max=448, err_mode="MAE", err_use_fast_math=False
            ),
            batch_invariant=True,
        )
