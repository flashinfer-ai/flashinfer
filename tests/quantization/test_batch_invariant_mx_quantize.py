# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""``flashinfer.mxfp8_quantize(input, ..., backend="auto", batch_invariant=True)`` and
``flashinfer.mxfp4_quantize(a, ..., backend="auto", batch_invariant=True)``: the classes ``mxfp8.quant`` and ``mxfp4.quant``
(TransformerEngine's MX recipe: the power-of-two scale of a 32-element block from its amax, rounded up, and its exact
inverse).

Device-free: the pinned fingerprints, the dispatcher's membership in the enumerated space and the device check. On an
SM100 / SM103 / SM107 device with the CuTe DSL: bit equality with the exact CPU oracle of the specification
(``quantization/mx_reference.py``: the vectorised model over a shape sweep and the Fractions evaluation of the symbolic
spec on a few rows) in the three scale-factor layouts and bf16 / f16 / fp32, with FlashInfer's MXFP8 reference
implementation (``tests/utils_fp8.py``, including ``alignment`` padding) and with FlashInfer's own quantizers (cuda and
cute-dsl) away from the one documented corner, which is pinned (a block whose amax is below the smallest scale: the
classes keep TE's inverse, FlashInfer's kernels zero the block); determinism including CUDA-graph replay and PDL;
configuration and SM-count independence; the batch-invariance guarantee itself (a row's bits do not depend on the batch it
is quantized in); the MX GEMMs consuming the outputs; M = 0 and the error contract. Every assertion is bitwise.
"""

import pytest
import torch

import flashinfer
from flashinfer import batch_invariant as bi
from flashinfer.batch_invariant.quantization import layout as L
from flashinfer.batch_invariant.quantization import mx_reference as R
from flashinfer.batch_invariant.quantization import mx_schedule as C
from flashinfer.batch_invariant.quantization import mx_spec as S
from flashinfer.batch_invariant.quantization.mx_api import (
    MXFP8_QUANT_ARCHS,
    mx_quantize,
    mxfp4_quantize_batch_invariant,
    mxfp8_quantize_batch_invariant,
)
from tests.test_helpers.batch_invariant_quant import (
    FI_LAYOUTS,
    adversarial_inputs,
    same_bits,
    varied_inputs,
)
from tests.utils_fp8 import assert_mxfp8_quantize_exact

BLOCK = 32
FMTS = [S.MXFP8, S.MXFP4]
FMT_IDS = ["mxfp8", "mxfp4"]
DT = [torch.bfloat16, torch.float16, torch.float32]
SWEEP = {
    torch.bfloat16: [
        (1, 32),
        (3, 96),
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
    (96, "128x4", 1),
    (1024, "128x4", 1),
    (4096, "128x4", 3),
    (1024, "8x4", 3),
    (1024, "linear", 3),
]
_supported = torch.cuda.is_available() and bi.is_supported_device(
    torch.device("cuda"), MXFP8_QUANT_ARCHS
)
needs_device = pytest.mark.skipif(
    not _supported,
    reason="needs a CUDA device of SM100 / SM103 / SM107 with the CuTe DSL",
)


def _quantize(x, fmt, layout="128x4", **kw):
    """The batch-invariant path through the public API; the codes as uint8 and the flat scale bytes, on the host."""
    if fmt is S.MXFP8:
        q, sf = flashinfer.mxfp8_quantize(
            x, sf_swizzle_layout=FI_LAYOUTS[layout], batch_invariant=True, **kw
        )
        assert q.dtype == torch.float8_e4m3fn
    else:
        q, sf = flashinfer.mxfp4_quantize(
            x, sfLayout=FI_LAYOUTS[layout], batch_invariant=True, **kw
        )
        assert q.dtype == torch.uint8
    assert sf.dtype == torch.uint8 and sf.dim() == 1
    return q.view(torch.uint8).cpu(), sf.cpu()


def _flashinfer(x, fmt, layout, backend):
    """FlashInfer's own quantizer (``backend`` cuda or cute-dsl) on the same inputs; the outputs on the host as bytes."""
    if fmt is S.MXFP8:
        q, sf = flashinfer.mxfp8_quantize(
            x, sf_swizzle_layout=FI_LAYOUTS[layout], backend=backend
        )
    else:
        q, sf = flashinfer.mxfp4_quantize(
            x, backend=backend, sfLayout=FI_LAYOUTS[layout]
        )
    return q.view(torch.uint8).cpu(), sf.view(torch.uint8).cpu().reshape(-1)


def _logical(outs, M, K, layout):
    """Per-row views whatever the storage layout: the code bytes ``(M, .)`` and the scale bytes ``(M, K / 32)``."""
    return outs[0].reshape(M, -1), L.unswizzle_sf(outs[1], M, K, BLOCK, layout)


# ----------------------------------------------------------------------------------------------- device-free


def test_fingerprints_are_pinned_and_separate_the_classes():
    """The receipts: one specification per format for every row count and row length (the layouts are placements of
    the store), so the fingerprints take no dimension."""
    assert bi.fingerprint("mxfp8_quantize") == "mxquant-mxfp8-9018bd1627b3e72a"
    assert bi.fingerprint("mxfp4_quantize") == "mxquant-mxfp4-af5e163620dac20c"
    assert bi.class_name("mxfp8_quantize") == "mxfp8.quant"
    assert bi.class_name("mxfp4_quantize") == "mxfp4.quant"
    with pytest.raises(TypeError):
        bi.fingerprint("mxfp8_quantize", K=4096)


def test_dispatcher_picks_are_members_of_the_space():
    """R3 membership: every configuration the closed-form dispatcher returns, for every layout, element width and the
    SM counts of the SM100 / SM103 / SM107 parts, is a member of the enumerated space the agreement test covers; the
    tile placements exist in the 128x4 layout only; more SMs never pick fewer CTAs."""
    for K in (32, 96, 1024, 4096, 16384):
        for eb in (2, 4):
            for layout in L.LAYOUTS:
                space = set(C.all_valid_configs(K, eb, layout))
                assert space
                for M in (1, 2, 16, 37, 128, 1024, 4096, 16384):
                    for sm in SM_COUNTS:
                        cfg = C.default_config(M, K, eb, sm, layout)
                        assert cfg in space, (K, eb, layout, sm, M, cfg.describe())
                        assert cfg.mode != "tile" or layout == "128x4"


def test_unsupported_device_raises_before_any_kernel():
    x = torch.zeros(4, 64, dtype=torch.bfloat16)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="mxfp8.quant"):
        mxfp8_quantize_batch_invariant(x)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="mxfp4.quant"):
        mxfp4_quantize_batch_invariant(x)


# ----------------------------------------------------------------------------------------------- on the device


@needs_device
@pytest.mark.parametrize("fmt", FMTS, ids=FMT_IDS)
@pytest.mark.parametrize("layout", L.LAYOUTS)
@pytest.mark.parametrize("dtype", DT)
def test_matches_exact_reference(dtype, layout, fmt):
    """Codes and scale bytes (the whole buffer: the padding bytes are zero) equal the exact CPU oracle over the shape
    sweep, with the recipe's corners in the inputs; a few rows equal the Fractions evaluation of the symbolic spec."""
    for M, K in SWEEP[dtype]:
        x = adversarial_inputs(M, K, dtype, seed=M + K, block=BLOCK)
        outs = _quantize(x, fmt, layout)
        assert outs[0].shape == (M, K * fmt.elem_bits // 8)
        assert outs[1].shape == (L.sf_numel(M, K, BLOCK, layout),)
        ref_codes, ref_sf, _ = R.quantize_mx_reference(x, fmt, swizzled=layout)
        assert torch.equal(outs[0], ref_codes), ("codes", M, K)
        assert torch.equal(outs[1], ref_sf), ("scales", M, K)
    rows = [0, 1, 4, 36]
    x = adversarial_inputs(37, 256, dtype, seed=7, block=BLOCK)
    got = _logical(_quantize(x, fmt, layout), 37, 256, layout)
    ex_codes, ex_sf = R.exact_rows(x, fmt, rows)
    assert torch.equal(got[0][rows], ex_codes) and torch.equal(got[1][rows], ex_sf)


@needs_device
@pytest.mark.parametrize("layout", L.LAYOUTS)
@pytest.mark.parametrize("dtype", DT)
def test_mxfp8_matches_flashinfer_reference_and_alignment(dtype, layout):
    """Codes and scale factors agree with FlashInfer's MXFP8 reference implementation over the shape sweep, and with
    ``alignment`` padding (zero columns under a zero scale byte)."""
    for M, K in SWEEP[dtype]:
        x = varied_inputs(M, K, dtype, seed=M + K)
        q, sf = flashinfer.mxfp8_quantize(
            x,
            sf_swizzle_layout=FI_LAYOUTS[layout],
            backend="auto",
            batch_invariant=True,
        )
        assert_mxfp8_quantize_exact(x, q, sf, sf_swizzle_layout=FI_LAYOUTS[layout])
    x = varied_inputs(37, 96, dtype, seed=1)
    q, sf = flashinfer.mxfp8_quantize(
        x, alignment=128, sf_swizzle_layout=FI_LAYOUTS[layout], batch_invariant=True
    )
    assert q.shape == (37, 128)
    assert_mxfp8_quantize_exact(
        x, q, sf, alignment=128, sf_swizzle_layout=FI_LAYOUTS[layout]
    )


@needs_device
@pytest.mark.parametrize("fmt", FMTS, ids=FMT_IDS)
@pytest.mark.parametrize("backend", ["cuda", "cute-dsl"])
def test_matches_flashinfer_backends(backend, fmt):
    """FlashInfer's own quantizers compute the classes on inputs without tiny blocks (the corner pinned below): codes and
    scale bytes equal in the three layouts; fp32 inputs on the MXFP8 CuTe-DSL backend (the others take 16-bit inputs)."""
    dtypes = [torch.bfloat16, torch.float16]
    if fmt is S.MXFP8 and backend == "cute-dsl":
        dtypes.append(torch.float32)
    for dtype in dtypes:
        for layout in L.LAYOUTS:
            for M, K in [(37, 256), (130, 1024), (3, 4096), (8, 96), (200, 7168)]:
                x = varied_inputs(M, K, dtype, seed=M)
                ours = _quantize(x, fmt, layout)
                theirs = _flashinfer(x, fmt, layout, backend)
                assert same_bits(ours, theirs), (dtype, layout, M, K)


@needs_device
@pytest.mark.parametrize("fmt", FMTS, ids=FMT_IDS)
def test_tiny_block_corner_is_pinned(fmt):
    """The one documented difference from FlashInfer's kernels: for a block whose amax is below ``elem_max x 2^-127``
    (scale byte 0) their inverse scale is 0 (every code 0), TE's and the classes' is 2^127 (the values survive). The
    scale bytes agree; every differing code sits in such a block; nothing else differs."""
    M, K = 64, 1024
    x = varied_inputs(M, K, torch.bfloat16, seed=11)
    xb = x.float().view(M, K // BLOCK, BLOCK)
    tiny = torch.zeros(M, K // BLOCK, dtype=torch.bool, device="cuda")
    tiny[::7, ::5] = True
    xb = torch.where(
        tiny.unsqueeze(-1), (xb.sign() + (xb == 0).float()) * 2.0**-126, xb
    )
    x = xb.view(M, K).to(torch.bfloat16)
    ours = _quantize(x, fmt, "128x4")
    theirs = _flashinfer(x, fmt, "128x4", "cuda")
    assert torch.equal(ours[1], theirs[1]), "scale bytes differ"
    sf = L.unswizzle_sf(ours[1], M, K, BLOCK, "128x4")
    assert bool((sf[tiny.cpu()] == 0).all())
    ov = R.unpack_elems(ours[0], fmt).view(M, K // BLOCK, BLOCK)
    tv = R.unpack_elems(theirs[0], fmt).view(M, K // BLOCK, BLOCK)
    diff = (ov != tv).any(-1)
    assert bool((tv[tiny.cpu()] == 0).all()) and bool((ov[tiny.cpu()] != 0).any())
    assert bool((diff <= tiny.cpu()).all()), "a code differs outside a tiny block"


@needs_device
@pytest.mark.parametrize("fmt", FMTS, ids=FMT_IDS)
def test_determinism_graph_replay_and_pdl(fmt):
    x = adversarial_inputs(37, 1024, torch.bfloat16, block=BLOCK)
    ref = _quantize(x, fmt, "128x4")
    for _ in range(5):
        assert same_bits(_quantize(x, fmt, "128x4"), ref)
    for pdl in (False, True):
        assert same_bits(_quantize(x, fmt, "128x4", enable_pdl=pdl), ref)
    api = flashinfer.mxfp8_quantize if fmt is S.MXFP8 else flashinfer.mxfp4_quantize
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        outs = api(x, batch_invariant=True)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            outs = api(x, batch_invariant=True)
    for _ in range(2):
        outs[0].view(torch.uint8).zero_()
        outs[1].zero_()
        g.replay()
        torch.cuda.synchronize()
        assert same_bits((outs[0].view(torch.uint8).cpu(), outs[1].cpu()), ref)


@needs_device
@pytest.mark.parametrize("fmt", FMTS, ids=FMT_IDS)
@pytest.mark.parametrize("layout", L.LAYOUTS)
def test_batch_permutation_subbatch_invariance(layout, fmt):
    """A row's bits do not depend on the batch it is quantized in. Against a batch of 300 rows: every row alone at the
    padding boundaries, rows embedded among unrelated rows and in a much larger batch, random orders and the reverse,
    sub-batches of every size class (random, contiguous, strided, with repeated rows). The sizes span the dispatcher's
    regimes, so different sub-batches run under different configurations."""
    K, M = 1024, 300
    g = torch.Generator().manual_seed(K)
    xb = adversarial_inputs(M, K, torch.bfloat16, seed=1, block=BLOCK)

    def rows(x):
        return _logical(_quantize(x.contiguous(), fmt, layout), x.shape[0], K, layout)

    def check(got, idx):
        idx = torch.as_tensor(idx)
        assert same_bits(got, tuple(t[idx] for t in full))

    full = rows(xb)
    for i in (0, 1, 7, 31, 127, 128, 129, 255, 256, 299):
        check(rows(xb[i : i + 1]), [i])
    other = adversarial_inputs(
        23, K, torch.bfloat16, seed=5, special=False, block=BLOCK
    )
    for pos, i in ((0, 17), (4, 130), (22, 299)):
        xo = other.clone()
        xo[pos] = xb[i]
        check(tuple(t[pos : pos + 1] for t in rows(xo)), [i])
    big = adversarial_inputs(
        2048, K, torch.bfloat16, seed=9, special=False, block=BLOCK
    )
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
@pytest.mark.parametrize("fmt", FMTS, ids=FMT_IDS)
@pytest.mark.parametrize("K,layout,stride", CONFIG_CASES)
def test_config_agreement(K, layout, stride, fmt):
    """R3: the members of the enumerated configuration space (elements per lane, blocks per thread, rows per CTA, the
    tile placements of the 128x4 layout) give the same bits; fp32 inputs over their space at one row length."""
    for dtype, eb in ((torch.bfloat16, 2), (torch.float32, 4)):
        if dtype == torch.float32 and (K != 1024 or layout != "128x4"):
            continue
        x = adversarial_inputs(37, K, dtype, seed=K, block=BLOCK)
        ref = tuple(o.cpu() for o in mx_quantize(x, fmt, layout))
        cfgs = C.all_valid_configs(K, eb, layout)
        picked = C.default_config(37, K, eb, C.DEFAULT_SM_COUNT, layout)
        assert picked in cfgs
        for cfg in cfgs[::stride] + [picked]:
            got = mx_quantize(x, fmt, layout, config=cfg)
            assert same_bits(tuple(o.cpu() for o in got), ref), (layout, cfg.describe())
        assert len(cfgs) >= 4


@needs_device
@pytest.mark.parametrize("fmt", FMTS, ids=FMT_IDS)
def test_sm_count_does_not_change_bits(fmt):
    """The dispatcher takes the device's SM count; its picks for the SM100 / SM103 / SM107 parts may differ, the bits
    never."""
    x = adversarial_inputs(2048, 4096, torch.bfloat16, seed=9, block=BLOCK)
    ref = tuple(o.cpu() for o in mx_quantize(x, fmt))
    picks = set()
    for sm in SM_COUNTS:
        assert same_bits(
            tuple(o.cpu() for o in mx_quantize(x, fmt, sm_count=sm)), ref
        ), sm
        picks.add(C.default_config(2048, 4096, 2, sm, "128x4").describe())
    print(f"{fmt.name}: dispatcher picks over {SM_COUNTS}: {sorted(picks)}")


@needs_device
@pytest.mark.parametrize("fmt", FMTS, ids=FMT_IDS)
def test_widenings_and_layouts_carry_the_same_bits(fmt):
    """One class per format: fp32 inputs give the bits of the 16-bit inputs they exactly widen, and the three layouts
    carry the same bytes (the layout is a placement of the store)."""
    for dtype in (torch.bfloat16, torch.float16):
        x = adversarial_inputs(130, 2048, dtype, seed=3, block=BLOCK)
        for layout in L.LAYOUTS:
            assert same_bits(
                _quantize(x.float(), fmt, layout), _quantize(x, fmt, layout)
            ), (dtype, layout)
    for M, K in ((300, 1024), (37, 96), (1000, 4096)):
        x = adversarial_inputs(M, K, torch.bfloat16, seed=M, block=BLOCK)
        ref = _logical(_quantize(x, fmt, "128x4"), M, K, "128x4")
        for layout in ("8x4", "linear"):
            assert same_bits(_logical(_quantize(x, fmt, layout), M, K, layout), ref), (
                layout
            )


@needs_device
@pytest.mark.parametrize("fmt", FMTS, ids=FMT_IDS)
def test_mx_gemm_consumes_the_outputs(fmt):
    """FlashInfer's MX GEMM backends that run here consume the classes' ``(codes, sf)`` (the layout is the one the GEMM
    reads): the product equals the same GEMM on FlashInfer's own quantization of the same inputs (byte-identical
    operands) and is close to the bf16 reference."""
    m, n, k = 128, 256, 2048
    a = (
        torch.randn(m, k, generator=torch.Generator().manual_seed(1))
        .to(torch.bfloat16)
        .cuda()
    )
    b = (
        torch.randn(n, k, generator=torch.Generator().manual_seed(2))
        .to(torch.bfloat16)
        .cuda()
    )
    ours = [_quantize(t, fmt, "128x4") for t in (a, b)]
    theirs = [_flashinfer(t, fmt, "128x4", "cuda") for t in (a, b)]
    for o, t in zip(ours, theirs, strict=True):
        assert same_bits(o, t)
    ref = a.double() @ b.double().T
    ran = 0
    for backend in (
        "cutlass",
        "cudnn",
        "cute-dsl",
    ):  # not trtllm: its weight is shuffled, another input

        def gemm(q):
            (aq, asf), (bq, bsf) = (tuple(t.cuda() for t in p) for p in q)
            if fmt is S.MXFP8:
                return flashinfer.mm_mxfp8(
                    aq.view(torch.float8_e4m3fn),
                    bq.view(torch.float8_e4m3fn).t(),
                    asf,
                    bsf,
                    out_dtype=torch.bfloat16,
                    backend=backend,
                )
            alpha = torch.ones(1, dtype=torch.float32, device="cuda")
            return flashinfer.mm_fp4(
                aq,
                bq.T,
                asf.view(-1, k // BLOCK),
                bsf.view(-1, k // BLOCK).T,
                alpha,
                torch.bfloat16,
                block_size=BLOCK,
                use_8x4_sf_layout=False,
                backend=backend,
                use_nvfp4=False,
            )

        try:
            out = gemm(ours)
        except Exception:  # noqa: BLE001  (the backend does not run this format / shape / device)
            continue
        ran += 1
        assert torch.equal(out, gemm(theirs)), backend
        cos = torch.nn.functional.cosine_similarity(
            out.double().flatten(), ref.flatten(), dim=0
        )
        assert float(cos) > 0.98, (backend, float(cos))
    assert ran >= 1


@needs_device
@pytest.mark.parametrize("fmt", FMTS, ids=FMT_IDS)
def test_zero_rows_and_unsupported_inputs(fmt):
    """M = 0 returns the documented shapes without a launch; a 3-D input keeps its leading shape; a view with an
    under-aligned row stride gives the bits of its contiguous copy; inputs outside the class raise
    BatchInvariantUnsupportedError through the public API (never an assertion, never a fallback)."""
    api = flashinfer.mxfp8_quantize if fmt is S.MXFP8 else flashinfer.mxfp4_quantize
    lay = "sf_swizzle_layout" if fmt is S.MXFP8 else "sfLayout"
    kbytes = 64 * fmt.elem_bits // 8
    for shape, layout in (((0, 64), "128x4"), ((0, 5, 96), "8x4"), ((0, 64), "linear")):
        x0 = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
        q, sf = api(x0, batch_invariant=True, **{lay: FI_LAYOUTS[layout]})
        assert q.shape == (
            *shape[:-1],
            shape[-1] * fmt.elem_bits // 8,
        ) and sf.shape == (0,)
    x3 = adversarial_inputs(36, 64, torch.bfloat16, seed=3, block=BLOCK).view(3, 12, 64)
    c = _quantize(x3, fmt)
    d = _quantize(x3.view(36, 64), fmt)
    assert (
        c[0].shape == (3, 12, kbytes)
        and torch.equal(c[0].view(36, kbytes), d[0])
        and torch.equal(c[1], d[1])
    )
    wide = varied_inputs(
        8, 4100, torch.bfloat16, seed=7
    )  # row stride 8200 bytes: not a multiple of 16
    view = wide[:, :4096]
    assert view.stride(0) * 2 % 16 != 0
    assert same_bits(_quantize(view, fmt), _quantize(view.contiguous(), fmt))
    x = varied_inputs(4, 1024, torch.bfloat16)
    E = bi.BatchInvariantUnsupportedError
    with pytest.raises(ValueError, match="requires backend='auto'"):
        api(x, backend="cuda", batch_invariant=True)
    with pytest.raises(E, match="multiple of 32"):
        api(x[:, :1008], batch_invariant=True)
    with pytest.raises(E, match="rows of at most"):
        api(
            torch.zeros(2, 65568, dtype=torch.bfloat16, device="cuda"),
            batch_invariant=True,
        )
    with pytest.raises(E, match="dtype"):
        api(x.to(torch.float8_e4m3fn), batch_invariant=True)
    with pytest.raises(E, match="CUDA device"):
        api(x.cpu(), batch_invariant=True)
    if fmt is S.MXFP8:
        with pytest.raises(E, match="alignment"):
            api(x, alignment=48, batch_invariant=True)
