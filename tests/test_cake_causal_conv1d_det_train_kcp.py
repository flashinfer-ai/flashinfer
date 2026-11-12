"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Single-GPU tests of ``cake_causal_conv1d_det_kcp`` (FlashInfer issue #6919): one process, world size 1, no process
group.  The rank holds every fragment of a partitioned problem (the halo exchange of one rank is the identity), so
the history rows, the backward's head rows and the tail fold all run through the host flow.  Criteria:

(a) ``y``, ``dx`` and ``dweight`` against an FP64 oracle of the KCP semantics (the SiLU derivative taken on the FP32
    preactivation, "policy B") within the conv package's contract tolerances: BF16 ``y`` / ``dx`` within
    ``atol 1e-3 + rtol 2^-7`` (one BF16 ulp) with no element outside, FP32 ``dweight`` within ``atol 5e-2 + rtol 1e-4``.
(b) against the shipped zero-history kernels on the unpartitioned documents: ``y`` bitwise equal (the forward runs the
    same instructions), and ``dx`` elementwise within ``1e-2 + 1e-2 |dx_noCP|`` with no mismatch.  The shipped backward
    differentiates the BF16-rounded preactivation ("policy A"), so its ``dx`` is not the KCP ``dx`` bit for bit: the
    two differ by the derivative's sensitivity to that rounding plus one BF16 output rounding (a few elements per
    million sit beyond ``2e-3 + 2^-6 |dx|``, none beyond the clause above; the measured difference is reported as a
    test property).
(c) bitwise: two autograd calls on the same inputs agree on ``y`` / ``dx`` / ``dweight``, the autograd function equals
    the direct forward / backward calls, and the fold of the received ``dpre`` rows reproduces the unpartitioned
    launch's ``dx`` through the same host flow.
(d) every output repeats bit for bit over ``REPEATS`` calls per launch form (``y``, ``hist``, ``dx``, ``dweight`` and
    the halo rows of both exchanges); a steady round issues no host-to-device copy, no synchronizing operation and no
    device allocation.
(e) the KCP backend reads no environment variable, and a copy of the package imports with ``flashinfer`` blocked.

(a) and (b) also run on the two production layers of FlashInfer issue #8365 (``ROWS_8365``: one rank's 32768 tokens at
C = 10240 and 16384 tokens at C = 8192).  A ``CakeKDADetKcpContext`` is built with
``cake_kda_det_train_kcp_build_context`` of the KDA KCP plan module.
"""

from __future__ import annotations

import types

import pytest
import torch

from flashinfer.cake_causal_conv1d_det_train import (
    cake_causal_conv1d_det_train_backward,
    cake_causal_conv1d_det_train_forward,
    is_cake_causal_conv1d_det_train_supported,
)
from flashinfer.experimental.cake.cake_causal_conv1d_det_train.kcp import (
    cake_causal_conv1d_det_kcp,
    cake_causal_conv1d_det_train_kcp_backward,
    cake_causal_conv1d_det_train_kcp_forward,
)

WIDTH = 4
HALO = WIDTH - 1
# The conv package's contract tolerances: y / dx within one BF16 ulp of the oracle (no element outside), dweight
# within atol 5e-2 + rtol 1e-4 of the FP64 one.
Y_ATOL, Y_RTOL = 1e-3, 2**-7
DW_ATOL, DW_RTOL = 5e-2, 1e-4
# The elementwise clause for a context-parallel result against the no-CP kernel of the same package.
NOCP_ATOL, NOCP_RTOL = 1e-2, 1e-2

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not is_cake_causal_conv1d_det_train_supported(torch.device("cuda")),
    reason="cake_causal_conv1d_det_train needs a supported CUDA device (SM100 / SM103 / SM107)",
)

# Rows: (document lengths, fragment lengths over the packed sequence); each fragment is cut at document
# boundaries too.  The rows of record C7 (history rows from two source fragments of one and two tokens, a head row
# with two consumers) and C7b (an empty predecessor), the [1 | 1 | 1 | long] pattern (a head row with three
# consumers), 1-3 token documents inside a fragment, a 1-token continuing fragment, and a wide row.
ROWS = [
    ("c7", (5000, 1003), (1, 2, 4997, 1003), 512),
    ("c7b", (5000, 1003), (1, 2, 0, 6000), 512),
    ("three-consumers", (1000, 0, 3, 3094), (1, 1, 1, 4094), 2048),
    ("short-docs", (1, 2, 3, 57), (3, 30, 30), 384),
    ("one-token-frag", (65,), (64, 1), 256),
    ("wide", (5000, 7000), (1, 2, 4997, 7000), 8192),
]
# One rank's tokens of the two production layers of FlashInfer issue #8365: 32768 tokens at C = 10240 (documents
# 12000 / 9000 / 6000 / 3768 / 2000) and 16384 tokens at C = 8192 (9000 / 5000 / 2384), cut into fragments whose
# boundaries are off multiples of 64 and inside documents, so the first document spans three fragments.  Kept out of
# ``ROWS`` (other suites parametrize over it); the FP64 oracle of a row holds several 2.7 GB tensors.
ROWS_8365 = [
    (
        "8365-c10240",
        (12000, 9000, 6000, 3768, 2000),
        (4097, 4095, 4101, 4091, 4096, 4096, 4093, 4099),
        10240,
    ),
    ("8365-c8192", (9000, 5000, 2384), (4095, 4097, 4099, 4093), 8192),
]
# Repeats of the determinism rows: every output, per row and launch form.
REPEATS = 5


def _fragments(lens, cuts):
    """``(doc, off, len)`` fragments in document-then-offset order, the cuts split at document boundaries; an empty
    cut is an empty fragment."""
    cu = [0]
    for n in lens:
        cu.append(cu[-1] + n)
    S = cu[-1]
    assert sum(cuts) == S, (cuts, S)
    frags = []
    pos = 0
    for n in cuts:
        end = pos + n
        if n == 0:
            d = (
                max(i for i in range(len(lens)) if cu[i] <= pos)
                if pos < S
                else len(lens) - 1
            )
            frags.append((d, pos - cu[d], 0))
        p = pos
        while p < end:
            d = max(i for i in range(len(lens)) if cu[i] <= p)
            e = min(end, cu[d + 1])
            frags.append((d, p - cu[d], e - p))
            p = e
        pos = end
    return S, tuple(cu), frags


def _context(lens, cuts, device):
    """A one-rank ``CakeKDADetKcpContext`` for ``cuts`` over documents ``lens``, built by the KDA KCP plan module (imported
    unconditionally: a renamed or moved plan fails here instead of silently testing a stand-in context)."""
    from flashinfer.experimental.cake.cake_kda_det_train_kcp.cake_kcp_plan import (
        cake_kda_det_train_kcp_build_context,
        cake_kda_det_train_kcp_partition,
    )

    S, cu_global, frags = _fragments(lens, cuts)
    part = cake_kda_det_train_kcp_partition(
        torch.tensor(cu_global, dtype=torch.int64), 1, fragments=[frags]
    )
    ctx = cake_kda_det_train_kcp_build_context(
        part, None, 0, device, conv_kernel_size=WIDTH
    )
    return ctx, S, cu_global, frags


def _problem(lens, cuts, C, device, seed=0):
    """The unpartitioned problem (``x``, ``w``, ``dy``, ``cu``) and the rank's local view of it: every fragment as a
    local sequence in the plan's order (``l2g`` maps local tokens to global ones)."""
    ctx, S, cu_global, frags = _context(lens, cuts, device)
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(S, C, device=device, generator=g, dtype=torch.bfloat16)
    w = (torch.randn(C, WIDTH, device=device, generator=g) * 0.5).contiguous()
    dy = torch.randn(S, C, device=device, generator=g, dtype=torch.bfloat16)
    l2g = []
    for d, off, ln in frags:
        l2g.extend(range(cu_global[d] + off, cu_global[d] + off + ln))
    l2g = torch.tensor(l2g, dtype=torch.int64, device=device)
    cu = torch.tensor(cu_global, dtype=torch.int32, device=device)
    return types.SimpleNamespace(
        ctx=ctx,
        S=S,
        C=C,
        x=x,
        w=w,
        dy=dy,
        cu=cu,
        l2g=l2g,
        x_local=x[l2g].contiguous(),
        dy_local=dy[l2g].contiguous(),
    )


def _reference(x, w, cu, dy):
    """FP64 oracle of the unpartitioned problem with the KCP semantics (policy B: the SiLU derivative on the
    unrounded preactivation): BF16 ``y`` and ``dx`` (one rounding each), FP64 ``dweight``."""
    S, C = x.shape
    xf = x.double()
    wd = w.double()
    cu_l = cu.tolist()
    doc = torch.zeros(S, dtype=torch.int64, device=x.device)
    for i in range(len(cu_l) - 1):
        doc[cu_l[i] : cu_l[i + 1]] = i

    def shifted(t, k):  # t[i - k] of the same document, else 0
        out = torch.zeros_like(t)
        if k < S:
            same = (
                doc[k:] == doc[:-k]
                if k
                else torch.ones(S, dtype=torch.bool, device=x.device)
            )
            out[k:] = torch.where(
                same[:, None], t[: S - k], torch.zeros_like(t[: S - k])
            )
        return out

    pre = sum(wd[:, k][None, :] * shifted(xf, HALO - k) for k in range(WIDTH))
    s = torch.sigmoid(pre)
    y = (pre * s).to(torch.bfloat16)
    dpre = dy.double() * s * (1 + pre * (1 - s))

    def shifted_up(t, k):  # t[i + k] of the same document, else 0
        out = torch.zeros_like(t)
        if k < S:
            same = (
                doc[k:] == doc[:-k]
                if k
                else torch.ones(S, dtype=torch.bool, device=x.device)
            )
            out[: S - k] = torch.where(same[:, None], t[k:], torch.zeros_like(t[k:]))
        return out

    dx = sum(wd[:, k][None, :] * shifted_up(dpre, HALO - k) for k in range(WIDTH)).to(
        torch.bfloat16
    )
    dw = torch.stack(
        [(dpre * shifted(xf, HALO - k)).sum(0) for k in range(WIDTH)],
        dim=1,
    )
    return y, dx, dw


def _assert_within(actual, expected, *, atol, rtol, label):
    """No element of ``actual`` outside ``atol + rtol |expected|`` (an exact mismatch count of zero); returns the
    comparison's statistics for the test record."""
    assert tuple(actual.shape) == tuple(expected.shape), label
    a = actual.detach().reshape(-1).double()
    b = expected.detach().reshape(-1).double()
    assert torch.isfinite(a).all(), f"{label} has non-finite values"
    diff = (a - b).abs()
    bound = atol + rtol * b.abs()
    outside = diff > bound
    count = int(outside.sum().item())
    stats = {
        "count_outside": count,
        "max_abs_diff": float(diff.max().item()),
        "max_ratio_to_bound": float((diff / bound).max().item()),
    }
    if count:
        index = int(outside.nonzero()[0].item())
        raise AssertionError(
            f"{label}: {count} of {a.numel()} elements outside atol={atol} rtol={rtol}; first at flat index "
            f"{index}: got {a[index].item()!r}, expected {b[index].item()!r}"
        )
    return stats


def _bits(t):
    return t.detach().view(torch.int16 if t.dtype == torch.bfloat16 else torch.int32)


@pytest.mark.parametrize(
    "name,lens,cuts,C", ROWS + ROWS_8365, ids=[r[0] for r in ROWS + ROWS_8365]
)
def test_forward_matches_unpartitioned_bitwise(name, lens, cuts, C):
    device = torch.device("cuda")
    p = _problem(lens, cuts, C, device)
    y_local, hist = cake_causal_conv1d_det_train_kcp_forward(p.x_local, p.w, p.ctx)
    y_global = cake_causal_conv1d_det_train_forward(p.x, p.w, p.cu)
    torch.cuda.synchronize()
    assert hist.shape[0] == max(1, p.ctx.halo.n_hist_rows)
    assert not y_local.isnan().any()
    # (b) the forward runs the zero-history forward's instructions with history rows: bitwise equal.
    assert torch.equal(_bits(y_local), _bits(y_global[p.l2g])), (
        "y differs from the zero-history kernel"
    )
    # (a) y against the FP64 oracle.
    y_ref, _, _ = _reference(p.x, p.w, p.cu, p.dy)
    _assert_within(
        y_local, y_ref[p.l2g], atol=Y_ATOL, rtol=Y_RTOL, label="y vs FP64 oracle"
    )


@pytest.mark.parametrize(
    "name,lens,cuts,C", ROWS + ROWS_8365, ids=[r[0] for r in ROWS + ROWS_8365]
)
def test_backward_matches_reference_and_folds_tails_bitwise(
    name, lens, cuts, C, record_property
):
    device = torch.device("cuda")
    p = _problem(lens, cuts, C, device)
    _, hist = cake_causal_conv1d_det_train_kcp_forward(p.x_local, p.w, p.ctx)
    dx_local, dw_local = cake_causal_conv1d_det_train_kcp_backward(
        p.x_local, p.w, p.ctx, p.dy_local, hist
    )
    torch.cuda.synchronize()
    assert not dx_local.isnan().any() and not dw_local.isnan().any()
    # (a) dx and dweight against the FP64 oracle of the KCP semantics.
    _, dx_ref, dw_ref = _reference(p.x, p.w, p.cu, p.dy)
    _assert_within(
        dx_local, dx_ref[p.l2g], atol=Y_ATOL, rtol=Y_RTOL, label="dx vs FP64 oracle"
    )
    _assert_within(
        dw_local, dw_ref, atol=DW_ATOL, rtol=DW_RTOL, label="dw vs FP64 oracle"
    )
    # (b) dx against the shipped zero-history backward (policy A) on the unpartitioned documents: the elementwise
    # clause for a context-parallel result against the no-CP kernel; the measured difference is recorded.
    dx_nocp, _ = cake_causal_conv1d_det_train_backward(p.x, p.w, p.cu, p.dy)
    stats = _assert_within(
        dx_local,
        dx_nocp[p.l2g],
        atol=NOCP_ATOL,
        rtol=NOCP_RTOL,
        label="dx vs no-CP kernel",
    )
    for key, value in stats.items():
        record_property(f"dx_vs_nocp_{key}", value)
    # (c) The unpartitioned problem through the same host flow (one fragment per document, no halo rows): the fold
    # of the received dpre rows reproduces its dx bit for bit.
    whole = _problem(lens, tuple(lens), C, device)
    _, hist_w = cake_causal_conv1d_det_train_kcp_forward(
        whole.x_local, whole.w, whole.ctx
    )
    dx_whole, _ = cake_causal_conv1d_det_train_kcp_backward(
        whole.x_local, whole.w, whole.ctx, whole.dy_local, hist_w
    )
    torch.cuda.synchronize()
    assert whole.ctx.halo.n_hist_rows == 0 and whole.ctx.halo.n_tail_rows == 0
    assert torch.equal(_bits(dx_local), _bits(dx_whole[whole.l2g.argsort()][p.l2g])), (
        "dx differs from the unpartitioned launch"
    )


@pytest.mark.parametrize("name,lens,cuts,C", ROWS[:3], ids=[r[0] for r in ROWS[:3]])
def test_autograd_function_and_repeat_are_bitwise(name, lens, cuts, C, record_property):
    device = torch.device("cuda")
    p = _problem(lens, cuts, C, device)
    runs = []
    for _ in range(2):
        x = p.x_local.clone().requires_grad_(True)
        w = p.w.clone().requires_grad_(True)
        y = cake_causal_conv1d_det_kcp(x, w, p.ctx)
        y.backward(p.dy_local)
        runs.append((y.detach(), x.grad, w.grad))
    y_direct, hist = cake_causal_conv1d_det_train_kcp_forward(p.x_local, p.w, p.ctx)
    dx_direct, dw_direct = cake_causal_conv1d_det_train_kcp_backward(
        p.x_local, p.w, p.ctx, p.dy_local, hist
    )
    torch.cuda.synchronize()
    (y1, dx1, dw1), (y2, dx2, dw2) = runs
    # (c) two launches on the same inputs, and the autograd function against the direct calls: bitwise.
    assert torch.equal(_bits(y1), _bits(y2))
    assert torch.equal(_bits(dx1), _bits(dx2))
    assert torch.equal(_bits(dw1), _bits(dw2)), "dweight differs between two launches"
    assert torch.equal(_bits(y1), _bits(y_direct))
    assert torch.equal(_bits(dx1), _bits(dx_direct))
    assert torch.equal(_bits(dw1), _bits(dw_direct)), (
        "autograd dweight differs from the direct backward"
    )
    # (a) the autograd outputs against the FP64 oracle of the KCP semantics.
    y_ref, dx_ref, dw_ref = _reference(p.x, p.w, p.cu, p.dy)
    _assert_within(y1, y_ref[p.l2g], atol=Y_ATOL, rtol=Y_RTOL, label="y vs FP64 oracle")
    _assert_within(
        dx1, dx_ref[p.l2g], atol=Y_ATOL, rtol=Y_RTOL, label="dx vs FP64 oracle"
    )
    _assert_within(dw1, dw_ref, atol=DW_ATOL, rtol=DW_RTOL, label="dw vs FP64 oracle")
    # (b) dx against the shipped zero-history backward (policy A) on the unpartitioned documents under the no-CP
    # clause; the two policies' difference is recorded for the report (it is not a bitwise relation).
    dx_nocp, _ = cake_causal_conv1d_det_train_backward(p.x, p.w, p.cu, p.dy)
    stats = _assert_within(
        dx1, dx_nocp[p.l2g], atol=NOCP_ATOL, rtol=NOCP_RTOL, label="dx vs no-CP kernel"
    )
    for key, value in stats.items():
        record_property(f"dx_vs_nocp_{key}", value)


def test_empty_rank_returns_empty_outputs():
    device = torch.device("cuda")
    C = 512
    ctx, S, _, _ = _context((0,), (0,), device)
    x = torch.empty(0, C, dtype=torch.bfloat16, device=device)
    w = torch.randn(C, WIDTH, device=device)
    y, hist = cake_causal_conv1d_det_train_kcp_forward(x, w, ctx)
    dx, dw = cake_causal_conv1d_det_train_kcp_backward(
        x, w, ctx, torch.empty_like(x), hist
    )
    assert y.shape == (0, C) and dx.shape == (0, C)
    assert torch.equal(dw, torch.zeros_like(dw))


def test_rejects_mismatched_context():
    device = torch.device("cuda")
    p = _problem((5000, 1003), (1, 2, 4997, 1003), 512, device)
    with pytest.raises(ValueError):
        cake_causal_conv1d_det_train_kcp_forward(
            p.x_local[:-1].contiguous(), p.w, p.ctx
        )


def test_kcp_keeps_fp32_taps_and_no_bias():
    """The KCP path keeps the package's pre-round-6 contract: BF16 taps are refused with the FP32 error text
    (the training routes accept them), and there is no bias keyword."""

    device = torch.device("cuda")
    p = _problem((5000, 1003), (1, 2, 4997, 1003), 512, device)
    with pytest.raises(ValueError, match="weight must be torch.float32"):
        cake_causal_conv1d_det_train_kcp_forward(
            p.x_local, p.w.to(torch.bfloat16), p.ctx
        )
    with pytest.raises(ValueError, match="weight must be torch.float32"):
        cake_causal_conv1d_det_train_kcp_backward(
            p.x_local,
            p.w.to(torch.bfloat16),
            p.ctx,
            p.dy_local,
            torch.empty(0, p.C, dtype=torch.bfloat16, device=device),
        )
    bias = torch.zeros(p.C, device=device)
    with pytest.raises(TypeError):
        cake_causal_conv1d_det_train_kcp_forward(p.x_local, p.w, p.ctx, bias=bias)
    with pytest.raises(TypeError):
        cake_causal_conv1d_det_kcp(p.x_local, p.w, p.ctx, bias)


def test_kcp_plans_with_the_legacy_tile_rule(monkeypatch):
    """The KCP programs plan their launches with ``cake_jit.plan(..., legacy=True)``: C = 1280 keeps ``(ct_vec, n_sub,
    n_ch_tiles) = (128, 1, 2)`` (the training routes tile it as (32, 4, 5)).  The backward does so on every form; so
    does the one-launch forward (knobs ``fwd_main`` 0, ``fwd_overlap`` 0), while the shipped forward's single-rank main
    program (``FWD_MAIN_ARCHS``) plans with the ``fwd`` route's rule for the device's architecture.  ``y`` stays
    bitwise the unpartitioned forward's either way."""

    from flashinfer.experimental.cake.cake_causal_conv1d_det_train import cake_jit
    from flashinfer.experimental.cake.cake_causal_conv1d_det_train import (
        kcp as conv_kcp,
    )

    geometry = {"threads": 128, "vec": 8}
    legacy = cake_jit.plan(5000, 1280, 2, geometry=geometry, legacy=True)
    assert (legacy["ct_vec"], legacy["n_sub"], legacy["n_ch_tiles"]) == (128, 1, 2)
    routed = cake_jit.plan(5000, 1280, 2, geometry=geometry, role="bwd")
    assert (routed["ct_vec"], routed["n_sub"], routed["n_ch_tiles"]) == (32, 4, 5)
    calls = []
    original = conv_kcp.plan

    def recording_plan(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(conv_kcp, "plan", recording_plan)
    device = torch.device("cuda")
    arch = conv_kcp._device_arch(device)
    p = _problem((5000, 1003), (1, 2, 4997, 1003), 1280, device)
    y_full = cake_causal_conv1d_det_train_forward(p.x, p.w, p.cu)
    for knobs in ({"fwd_main": 0, "fwd_overlap": 0}, None):
        calls.clear()
        # the trimmed host path caches its plans per context: each form plans afresh
        conv_kcp.cake_causal_conv1d_det_train_kcp_host_cache_clear()
        conv_kcp._set_test_knobs(knobs)
        try:
            y, hist = cake_causal_conv1d_det_train_kcp_forward(p.x_local, p.w, p.ctx)
            dx, dw = cake_causal_conv1d_det_train_kcp_backward(
                p.x_local, p.w, p.ctx, p.dy_local, hist
            )
        finally:
            conv_kcp._set_test_knobs(None)
        assert len(calls) == 2 and calls[1].get("legacy") is True, (knobs, calls)
        if knobs is None and arch in conv_kcp.FWD_MAIN_ARCHS:
            assert calls[0].get("legacy") is not True, calls
            assert (calls[0].get("route"), calls[0].get("arch")) == ("fwd", arch), calls
        else:
            assert calls[0].get("legacy") is True, (knobs, calls)
        assert torch.equal(y, y_full[p.l2g]), knobs
        assert torch.isfinite(dw).all() and dx.shape == p.x_local.shape


@pytest.mark.parametrize("host_trim", [0, 1])
def test_forward_grid_is_one_sm_short_unless_a_channel_tile_needs_the_sm(
    monkeypatch, host_trim
) -> None:
    """On both host paths the one-launch forward (knobs ``fwd_main`` 0, ``fwd_overlap`` 0: the ``kcp_fwd`` program)
    launches ``ctas_per_sm x (SMs - FWD_RESERVE_SMS)`` CTAs when that grid holds every channel tile of ``C``, else the
    record's full co-resident grid (``kcp._forward_grid``): with the device's full grid made exactly ``C``'s channel
    tile count (``_record_launch_grid`` patched), the forward launches that full grid instead of refusing ``C``, and
    ``y`` stays bitwise the unpartitioned forward's in both cases; a full grid one CTA short of ``C``'s tile count
    refuses ``C``."""

    from flashinfer.experimental.cake.cake_causal_conv1d_det_train import (
        kcp as conv_kcp,
    )

    device = torch.device("cuda")
    (name,) = conv_kcp.route_modules(conv_kcp._device_arch(device), conv_kcp.KCP_FWD)
    record = conv_kcp._loader.MODULES[name]
    ctas_per_sm = int(record["grid_rule"]["ctas_per_sm"])
    geometry = record["grid_rule"]["geometry"]
    reserved = conv_kcp.FWD_RESERVE_SMS * ctas_per_sm

    def tiles(C: int) -> int:
        return conv_kcp.plan(1, C, 1, geometry=geometry, legacy=True)["n_ch_tiles"]

    # the smallest C (a multiple of 1024) with more channel tiles than the reserved CTAs: a full grid of exactly its
    # tile count then leaves a shorter grid of at least one CTA, too small for C
    C = 1024
    while tiles(C) <= reserved:
        C += 1024
    n_ch_tiles = tiles(C)
    grids = []
    run, launch_plan = conv_kcp.run_module, conv_kcp._launch_plan

    def recording_run(module, values, **kwargs):
        if module == name:
            grids.append(tuple(kwargs["grid"]))
        return run(module, values, **kwargs)

    def recording_launch_plan(module, grid, device_index):
        if module == name:
            grids.append(tuple(grid))
        return launch_plan(module, grid, device_index)

    monkeypatch.setattr(conv_kcp, "run_module", recording_run)
    monkeypatch.setattr(conv_kcp, "_launch_plan", recording_launch_plan)
    device_full = conv_kcp._record_launch_grid(
        record, device_index=conv_kcp._device_index(device)
    )
    conv_kcp._set_test_knobs({"host_trim": host_trim, "fwd_main": 0, "fwd_overlap": 0})
    try:
        for full, launched in (
            (device_full, (device_full[0] - reserved,) + tuple(device_full[1:])),
            ((n_ch_tiles, 1, 1), (n_ch_tiles, 1, 1)),
        ):
            assert full[0] - reserved < n_ch_tiles or full is device_full
            monkeypatch.setattr(
                conv_kcp, "_record_launch_grid", lambda record, _full=full, **_kw: _full
            )
            grids.clear()
            p = _problem((5000, 1003), (1, 2, 4997, 1003), C, device)
            y, _hist = cake_causal_conv1d_det_train_kcp_forward(p.x_local, p.w, p.ctx)
            assert grids == [launched], (host_trim, full, grids)
            y_full = cake_causal_conv1d_det_train_forward(p.x, p.w, p.cu)
            assert torch.equal(y, y_full[p.l2g]), (host_trim, full)
        monkeypatch.setattr(
            conv_kcp,
            "_record_launch_grid",
            lambda record, **_kw: (n_ch_tiles - 1, 1, 1),
        )
        p = _problem((5000, 1003), (1, 2, 4997, 1003), C, device)
        with pytest.raises(ValueError, match="spans more channel tiles"):
            cake_causal_conv1d_det_train_kcp_forward(p.x_local, p.w, p.ctx)
    finally:
        conv_kcp._set_test_knobs(None)


@pytest.mark.parametrize("form", ["knob", "keyword"])
def test_graph_replay_knob_is_bitwise(form) -> None:
    """The graph-replay option: under ``_set_test_knobs({"graph": "replay"})`` (form ``knob``) or the public keyword
    ``graph="replay"`` (form ``keyword``) the first call of a key runs eagerly,
    the second is captured and replayed once, later calls replay the graph; ``y`` / ``hist`` / ``dx`` / ``dweight``
    are bitwise the knob-off results on every call.  The backward's key holds the forward's ``hist`` pointer, so
    its capture follows the forward's by one call.  Another input pointer is another key (eager, still bitwise)."""
    from flashinfer.experimental.cake.cake_causal_conv1d_det_train import (
        kcp as conv_kcp,
    )

    device = torch.device("cuda")
    p = _problem((5000, 1003), (1, 2, 4997, 1003), 512, device)
    x, w, ctx = p.x_local, p.w, p.ctx
    dy = torch.randn(x.shape, device=device).to(torch.bfloat16)
    y0, hist0 = cake_causal_conv1d_det_train_kcp_forward(x, w, ctx)
    dx0, dw0 = cake_causal_conv1d_det_train_kcp_backward(x, w, ctx, dy, hist0)
    kw = {}
    if form == "knob":
        conv_kcp._set_test_knobs({"graph": "replay"})  # the private test hook
    else:
        kw = {"graph": "replay"}  # the public keyword
    try:
        events = []
        for _ in range(4):
            y, hist = cake_causal_conv1d_det_train_kcp_forward(x, w, ctx, **kw)
            events.append(conv_kcp.LAST_CALL["fwd"]["graph_event"])
            assert torch.equal(y, y0) and torch.equal(hist, hist0)
            dx, dw = cake_causal_conv1d_det_train_kcp_backward(
                x, w, ctx, dy, hist, **kw
            )
            events.append(conv_kcp.LAST_CALL["bwd"]["graph_event"])
            assert torch.equal(dx, dx0) and torch.equal(dw, dw0)
        assert events == [
            "eager",
            "eager",
            "capture",
            "eager",
            "replay",
            "capture",
            "replay",
            "replay",
        ], events
        assert conv_kcp.GRAPH_STATS["fallback"] == 0
        y, hist = cake_causal_conv1d_det_train_kcp_forward(x.clone(), w, ctx, **kw)
        assert conv_kcp.LAST_CALL["fwd"]["graph_event"] == "eager"
        assert torch.equal(y, y0) and torch.equal(hist, hist0)
        if (
            form == "keyword"
        ):  # the autograd entry carries the keyword to the forward and to the backward
            xg = x.clone().requires_grad_(True)
            y = conv_kcp.cake_causal_conv1d_det_kcp(xg, w, ctx, graph="replay")
            assert conv_kcp.LAST_CALL["fwd"]["graph"] == "replay"
            assert torch.equal(y.detach(), y0)
            y.backward(dy)
            assert conv_kcp.LAST_CALL["bwd"]["graph"] == "replay"
            assert torch.equal(xg.grad, dx0)
    finally:
        conv_kcp._set_test_knobs(None)
        conv_kcp._GRAPHS.clear()
    y, hist = cake_causal_conv1d_det_train_kcp_forward(x, w, ctx)
    assert torch.equal(y, y0) and torch.equal(hist, hist0)


# ----------------------------------------------------------------------------------------------------------------
# Recompute handle (issues #7815 / #7818): torch's checkpoint recompute binds the history pool the original forward
# stored instead of gathering and exchanging it again.  One rank here (the exchange of one rank is the identity, so
# the module's ``_exchange`` calls are counted instead of collectives: one in the original forward, one in the
# backward, none in a recompute with a handle); the NCCL rows are in test_cake_causal_conv1d_det_train_kcp_dist.py.


class _ExchangeCounter:
    """Counts ``kcp._exchange`` calls (the halo exchange of the forward and of the backward; nothing else) per
    phase: ``orig``, ``recompute`` (set by the checkpoint's ``context_fn``, re-entered per recompute) and ``bwd``."""

    def __init__(self, monkeypatch, conv_kcp) -> None:
        self.phase = "orig"
        self.calls: dict[str, int] = {}
        self._stack: list[str] = []
        original = conv_kcp._exchange

        def counting(*a, **kw):
            self.calls[self.phase] = self.calls.get(self.phase, 0) + 1
            return original(*a, **kw)

        monkeypatch.setattr(conv_kcp, "_exchange", counting)

    def recompute_cm(self):
        counter = self

        class _Phase:
            def __enter__(self):
                counter._stack.append(counter.phase)
                counter.phase = "recompute"
                return self

            def __exit__(self, *exc):
                counter.phase = counter._stack.pop()
                return False

        return _Phase()

    def table(self) -> dict[str, int]:
        return {p: self.calls.get(p, 0) for p in ("orig", "recompute", "bwd")}


def _ckpt_run(p, *, handle=None, reentrant=False, ckpt=True, counter=None, twice=False):
    """``y`` / ``dx`` / ``dweight`` of ``cake_causal_conv1d_det_kcp`` (the facade) on the problem ``p``, directly or under a
    checkpoint, with ``recompute_handle=handle``."""
    import contextlib

    from torch.utils.checkpoint import checkpoint

    from flashinfer import cake_causal_conv1d_det_train_kcp as facade

    x = p.x_local.clone().requires_grad_(True)
    w = p.w.clone().requires_grad_(True)

    def fn(x_, w_, h_):
        return facade.cake_causal_conv1d_det_kcp(x_, w_, p.ctx, recompute_handle=h_)

    if not ckpt:
        y = fn(x, w, handle)
    elif reentrant:
        y = checkpoint(fn, x, w, handle, use_reentrant=True)
    else:
        cm = counter.recompute_cm if counter is not None else contextlib.nullcontext
        y = checkpoint(
            fn,
            x,
            w,
            handle,
            use_reentrant=False,
            context_fn=lambda: (contextlib.nullcontext(), cm()),
        )
    if counter is not None:
        counter.phase = "bwd"
    y.backward(p.dy_local, retain_graph=twice)
    if twice:
        y.backward(p.dy_local)
    return {"y": y.detach(), "dx": x.grad, "dw": w.grad}


def _same(a, b) -> bool:
    return all(torch.equal(_bits(a[n]), _bits(b[n])) for n in ("y", "dx", "dw"))


@pytest.mark.parametrize("name,lens,cuts,C", ROWS, ids=[r[0] for r in ROWS])
@pytest.mark.parametrize("reentrant", [False, True], ids=["nonreentrant", "reentrant"])
def test_recompute_handle_is_bitwise_and_exchange_free(
    monkeypatch, name, lens, cuts, C, reentrant
) -> None:
    """The original forward stores ``hist`` ("capture"), the recompute binds it ("replay") without an exchange;
    ``y`` / ``dx`` / ``dweight`` are bitwise the plain checkpoint's, the direct pass's and the explicit
    forward / backward calls' (``recompute_handle=None`` is the unchanged path); the handle holds exactly the
    history pool, the saved set's own tensor."""
    from flashinfer.experimental.cake.cake_causal_conv1d_det_train import (
        kcp as conv_kcp,
    )

    device = torch.device("cuda")
    p = _problem(lens, cuts, C, device)
    y_e, hist_e = cake_causal_conv1d_det_train_kcp_forward(p.x_local, p.w, p.ctx)
    dx_e, dw_e = cake_causal_conv1d_det_train_kcp_backward(
        p.x_local, p.w, p.ctx, p.dy_local, hist_e
    )
    explicit = {"y": y_e, "dx": dx_e, "dw": dw_e}
    direct = _ckpt_run(p, ckpt=False)
    plain_cc = _ExchangeCounter(monkeypatch, conv_kcp)
    plain = _ckpt_run(p, reentrant=reentrant, counter=plain_cc)
    monkeypatch.undo()
    handle = conv_kcp.CakeCausalConv1dDetKcpRecomputeHandle()
    cc = _ExchangeCounter(monkeypatch, conv_kcp)
    conv_kcp.LAST_CALL.pop("recompute_handle", None)
    got = _ckpt_run(p, handle=handle, reentrant=reentrant, counter=cc)
    assert conv_kcp.LAST_CALL["recompute_handle"] == {"fwd": "replay"}
    assert _same(direct, explicit), (
        "recompute_handle=None differs from the explicit calls"
    )
    assert _same(plain, direct), "plain checkpoint differs from the direct pass"
    assert _same(got, plain), (
        "checkpoint with a handle differs from the plain checkpoint"
    )
    if (
        reentrant
    ):  # the recompute runs inside the backward without a context_fn: counted as "bwd"
        assert plain_cc.table() == {"orig": 1, "recompute": 0, "bwd": 2}
        assert cc.table() == {"orig": 1, "recompute": 0, "bwd": 1}
    else:
        assert plain_cc.table() == {"orig": 1, "recompute": 1, "bwd": 1}
        assert cc.table() == {"orig": 1, "recompute": 0, "bwd": 1}
    entry = handle.entries[conv_kcp.HANDLE_KEY]
    hist = entry["hist"]
    assert hist.dtype == torch.bfloat16 and hist.shape == hist_e.shape
    assert torch.equal(hist, hist_e)
    assert handle.nbytes() == hist.numel() * hist.element_size()
    assert handle.nbytes() == int(p.ctx.halo.n_hist_rows) * C * 2
    handle.clear()
    assert handle.nbytes() == 0 and not handle.entries


def test_recompute_handle_retain_graph_and_nested_checkpoints(monkeypatch) -> None:
    """``retain_graph`` + a second backward replays the entry again (no exchange) and accumulates exactly twice;
    nested checkpoints with the handle created outside both regions never exchange in a recompute; a handle created
    inside the outer region is empty in the outer recompute, which then exchanges once -- still bitwise."""
    import contextlib

    from torch.utils.checkpoint import checkpoint

    from flashinfer import cake_causal_conv1d_det_train_kcp as facade
    from flashinfer.experimental.cake.cake_causal_conv1d_det_train import (
        kcp as conv_kcp,
    )

    device = torch.device("cuda")
    p = _problem((5000, 1003), (1, 2, 4997, 1003), 512, device)
    direct = _ckpt_run(p, ckpt=False)
    cc = _ExchangeCounter(monkeypatch, conv_kcp)
    twice = _ckpt_run(
        p,
        handle=conv_kcp.CakeCausalConv1dDetKcpRecomputeHandle(),
        counter=cc,
        twice=True,
    )
    assert cc.table() == {"orig": 1, "recompute": 0, "bwd": 2}
    assert torch.equal(_bits(twice["dx"]), _bits(direct["dx"] + direct["dx"]))
    assert torch.equal(_bits(twice["dw"]), _bits(direct["dw"] + direct["dw"]))
    monkeypatch.undo()

    def nested(inside: bool):
        counter = _ExchangeCounter(monkeypatch, conv_kcp)
        x = p.x_local.clone().requires_grad_(True)
        w = p.w.clone().requires_grad_(True)

        def inner(x_, w_, h_):
            return facade.cake_causal_conv1d_det_kcp(x_, w_, p.ctx, recompute_handle=h_)

        def outer(x_, w_, h_):
            hh = conv_kcp.CakeCausalConv1dDetKcpRecomputeHandle() if inside else h_
            yi = checkpoint(inner, x_, w_, hh, use_reentrant=False)
            return yi * torch.ones_like(
                yi
            )  # saves a tensor of the outer region; no bit changes

        y = checkpoint(
            outer,
            x,
            w,
            conv_kcp.CakeCausalConv1dDetKcpRecomputeHandle(),
            use_reentrant=False,
            context_fn=lambda: (contextlib.nullcontext(), counter.recompute_cm()),
        )
        counter.phase = "bwd"
        y.backward(p.dy_local)
        table = counter.table()
        monkeypatch.undo()
        return {"y": y.detach(), "dx": x.grad, "dw": w.grad}, table

    out_res, out_table = nested(False)
    in_res, in_table = nested(True)
    assert _same(out_res, direct) and _same(in_res, direct)
    assert out_table == {"orig": 1, "recompute": 0, "bwd": 1}, out_table
    assert in_table == {"orig": 1, "recompute": 1, "bwd": 1}, in_table


def test_recompute_handle_misuse_raises() -> None:
    """A forward outside the backward that finds an entry raises (a handle reused across micro-steps or calls);
    ``graph="replay"`` (keyword or knob) and a CUDA-graph capture are refused with a handle; an object without an
    ``entries`` dict is refused; a recompute whose entry describes another context or problem raises; a handle that
    lost its entry before the recompute makes it an original forward again (exchanges, still bitwise)."""
    from torch.utils.checkpoint import checkpoint

    from flashinfer import cake_causal_conv1d_det_train_kcp as facade
    from flashinfer.experimental.cake.cake_causal_conv1d_det_train import (
        kcp as conv_kcp,
    )

    device = torch.device("cuda")
    p = _problem((5000, 1003), (1, 2, 4997, 1003), 512, device)
    handle = conv_kcp.CakeCausalConv1dDetKcpRecomputeHandle()
    facade.cake_causal_conv1d_det_kcp(p.x_local, p.w, p.ctx, recompute_handle=handle)
    assert conv_kcp.LAST_CALL["recompute_handle"] == {"fwd": "capture"}
    with pytest.raises(RuntimeError, match="per call and micro-step"):
        facade.cake_causal_conv1d_det_kcp(
            p.x_local, p.w, p.ctx, recompute_handle=handle
        )
    with pytest.raises(ValueError, match="graph='off'"):
        conv_kcp.cake_causal_conv1d_det_kcp(
            p.x_local,
            p.w,
            p.ctx,
            graph="replay",
            recompute_handle=conv_kcp.CakeCausalConv1dDetKcpRecomputeHandle(),
        )
    with pytest.raises(TypeError, match="entries"):
        facade.cake_causal_conv1d_det_kcp(
            p.x_local, p.w, p.ctx, recompute_handle=object()
        )
    graph = torch.cuda.CUDAGraph()
    with pytest.raises(ValueError, match="capture"), torch.cuda.graph(graph):
        conv_kcp.cake_causal_conv1d_det_kcp(
            p.x_local,
            p.w,
            p.ctx,
            recompute_handle=conv_kcp.CakeCausalConv1dDetKcpRecomputeHandle(),
        )
    del graph
    # the public autograd Function refuses like the wrapper (graph keyword, capture, no entries dict)
    fresh = conv_kcp.CakeCausalConv1dDetKcpRecomputeHandle()
    with pytest.raises(ValueError, match="graph='off'"):
        conv_kcp.CakeCausalConv1dDetKcpFunction.apply(
            p.x_local, p.w, p.ctx, "replay", fresh
        )
    with pytest.raises(TypeError, match="entries"):
        conv_kcp.CakeCausalConv1dDetKcpFunction.apply(
            p.x_local, p.w, p.ctx, None, object()
        )
    graph = torch.cuda.CUDAGraph()
    with pytest.raises(ValueError, match="capture"), torch.cuda.graph(graph):
        conv_kcp.CakeCausalConv1dDetKcpFunction.apply(
            p.x_local, p.w, p.ctx, None, fresh
        )
    del graph
    assert not fresh.entries, "a refused call stored an entry"
    # a recompute whose entry belongs to another context / problem raises inside the backward
    other = _problem((5000, 1003), (1, 2, 4997, 1003), 512, device)
    for tamper in ("ctx", "key"):
        h = conv_kcp.CakeCausalConv1dDetKcpRecomputeHandle()
        x = p.x_local.clone().requires_grad_(True)

        def fn(x_, h_):
            return facade.cake_causal_conv1d_det_kcp(
                x_, p.w, p.ctx, recompute_handle=h_
            )

        y = checkpoint(fn, x, h, use_reentrant=False)
        if tamper == "ctx":
            h.entries["conv"]["ctx"] = other.ctx
        else:
            h.entries["conv"]["key"] = (1, 1)
        with pytest.raises(ValueError, match="recompute_handle"):
            y.backward(p.dy_local)
    # the entry dropped before the recompute: the recompute is an original forward (exchanges and stores)
    direct = _ckpt_run(p, ckpt=False)
    h = conv_kcp.CakeCausalConv1dDetKcpRecomputeHandle()
    x = p.x_local.clone().requires_grad_(True)
    w = p.w.clone().requires_grad_(True)
    y = checkpoint(
        lambda x_, w_, h_: facade.cake_causal_conv1d_det_kcp(
            x_, w_, p.ctx, recompute_handle=h_
        ),
        x,
        w,
        h,
        use_reentrant=False,
    )
    h.clear()
    y.backward(p.dy_local)
    assert conv_kcp.LAST_CALL["recompute_handle"] == {"fwd": "capture"}
    assert "conv" in h.entries
    assert _same({"y": y.detach(), "dx": x.grad, "dw": w.grad}, direct)


def test_recompute_handle_knob_graph_is_refused() -> None:
    from flashinfer.experimental.cake.cake_causal_conv1d_det_train import (
        kcp as conv_kcp,
    )

    device = torch.device("cuda")
    p = _problem((65,), (64, 1), 256, device)
    conv_kcp._set_test_knobs({"graph": "replay"})
    handle = conv_kcp.CakeCausalConv1dDetKcpRecomputeHandle()
    try:
        with pytest.raises(ValueError, match="graph='off'"):
            conv_kcp.cake_causal_conv1d_det_kcp(
                p.x_local, p.w, p.ctx, recompute_handle=handle
            )
        # the public autograd Function refuses the knob too
        with pytest.raises(ValueError, match="graph='off'"):
            conv_kcp.CakeCausalConv1dDetKcpFunction.apply(
                p.x_local, p.w, p.ctx, None, handle
            )
    finally:
        conv_kcp._set_test_knobs(None)
        conv_kcp._GRAPHS.clear()
    assert not handle.entries, "a refused call stored an entry"


@pytest.mark.parametrize("host_trim", [0, 1])
def test_recompute_handle_both_host_paths(monkeypatch, host_trim) -> None:
    """The replay binds the stored pool on the trimmed and on the previous host path alike: bitwise, and no exchange
    in the recompute (the plain checkpoint, the positive control, exchanges once there)."""
    from flashinfer.experimental.cake.cake_causal_conv1d_det_train import (
        kcp as conv_kcp,
    )

    device = torch.device("cuda")
    p = _problem((1, 2, 3, 57), (3, 30, 30), 384, device)
    conv_kcp._set_test_knobs({"host_trim": host_trim})
    exchange = conv_kcp._exchange
    try:
        direct = _ckpt_run(p, ckpt=False)
        plain_cc = _ExchangeCounter(monkeypatch, conv_kcp)
        plain = _ckpt_run(p, counter=plain_cc)
        # the next counter wraps the module's own exchange, not the first counter
        monkeypatch.setattr(conv_kcp, "_exchange", exchange)
        cc = _ExchangeCounter(monkeypatch, conv_kcp)
        got = _ckpt_run(
            p, handle=conv_kcp.CakeCausalConv1dDetKcpRecomputeHandle(), counter=cc
        )
        assert conv_kcp.LAST_CALL["recompute_handle"] == {"fwd": "replay"}
    finally:
        conv_kcp._set_test_knobs(None)
    assert _same(plain, direct) and _same(got, direct)
    assert plain_cc.table() == {"orig": 1, "recompute": 1, "bwd": 1}, plain_cc.table()
    assert cc.table() == {"orig": 1, "recompute": 0, "bwd": 1}, cc.table()


# ----------------------------------------------------------------------------------------------------------------
# The backward's launch forms (FlashInfer issue #8365): ``cooperative=True`` runs the cooperative ``kcp_bwd`` record,
# ``cooperative=False`` the ``kcp_bwd_split`` record (an ordinary launch one SM short of co-resident) and its fixed-order
# reduce program.  ``dx`` is bitwise the same in both forms; ``dweight`` is reproducible in each form and within the
# FP64-oracle bound, its bits differ between the forms (each sums its own partial rows).


class _RouteRecorder:
    """Records the backward kinds the host flow selects (``kcp.selected_modules``)."""

    def __init__(self, monkeypatch, conv_kcp) -> None:
        self.kinds: list[str] = []
        original = conv_kcp.selected_modules

        def recording(arch, kind, **kwargs):
            if kind in ("kcp_bwd", "kcp_bwd_split"):
                self.kinds.append(kind)
            return original(arch, kind, **kwargs)

        monkeypatch.setattr(conv_kcp, "selected_modules", recording)


@pytest.mark.parametrize(
    "name,lens,cuts,C", [*ROWS[:3], ROWS[5]], ids=[r[0] for r in [*ROWS[:3], ROWS[5]]]
)
@pytest.mark.parametrize("host_trim", [0, 1])
def test_backward_launch_forms(monkeypatch, name, lens, cuts, C, host_trim) -> None:
    from flashinfer.experimental.cake.cake_causal_conv1d_det_train import (
        kcp as conv_kcp,
    )

    device = torch.device("cuda")
    p = _problem(lens, cuts, C, device)
    conv_kcp._set_test_knobs({"host_trim": host_trim})
    conv_kcp.cake_causal_conv1d_det_train_kcp_host_cache_clear()
    routes = _RouteRecorder(monkeypatch, conv_kcp)
    try:
        _, hist = cake_causal_conv1d_det_train_kcp_forward(p.x_local, p.w, p.ctx)
        out = {}
        for coop in (True, False):
            runs = [
                cake_causal_conv1d_det_train_kcp_backward(
                    p.x_local, p.w, p.ctx, p.dy_local, hist, cooperative=coop
                )
                for _ in range(3)
            ]
            out[coop] = runs
        default = cake_causal_conv1d_det_train_kcp_backward(
            p.x_local, p.w, p.ctx, p.dy_local, hist
        )
        torch.cuda.synchronize()
    finally:
        conv_kcp._set_test_knobs(None)
    expected = (
        ["kcp_bwd"] * 3
        + ["kcp_bwd_split"] * 3
        + [conv_kcp._bwd_kind(conv_kcp.COOPERATIVE_DEFAULT)]
    )
    if host_trim == 1:  # the trimmed path selects once per cache key
        expected = ["kcp_bwd", "kcp_bwd_split"]
    assert routes.kinds == expected, routes.kinds
    _, dx_ref, dw_ref = _reference(p.x, p.w, p.cu, p.dy)
    dx_coop, dw_coop = out[True][0]
    for coop, runs in out.items():
        dx0, dw0 = runs[0]
        for dx, dw in runs[1:]:
            assert torch.equal(_bits(dx), _bits(dx0)) and torch.equal(
                _bits(dw), _bits(dw0)
            ), f"cooperative={coop}: a repeat differs"
        assert torch.equal(_bits(dx0), _bits(dx_coop)), (
            f"cooperative={coop}: dx differs from the cooperative form"
        )
        _assert_within(
            dw0, dw_ref, atol=DW_ATOL, rtol=DW_RTOL, label=f"dw (cooperative={coop})"
        )
        _assert_within(
            dx0,
            dx_ref[p.l2g],
            atol=Y_ATOL,
            rtol=Y_RTOL,
            label=f"dx (cooperative={coop})",
        )
    dx_d, dw_d = default
    dx_e, dw_e = out[conv_kcp.COOPERATIVE_DEFAULT][0]
    assert torch.equal(_bits(dx_d), _bits(dx_e)) and torch.equal(
        _bits(dw_d), _bits(dw_e)
    ), "cooperative=None is not the package default"


def test_backward_launch_form_autograd_graph_and_refusal(monkeypatch) -> None:
    """The autograd entry carries ``cooperative`` to its backward (bitwise the direct backward of that form), a graph
    replay of either form is bitwise its eager result (the form is part of the graph key), and a non-bool is
    refused."""
    from flashinfer.experimental.cake.cake_causal_conv1d_det_train import (
        kcp as conv_kcp,
    )

    device = torch.device("cuda")
    p = _problem((5000, 1003), (1, 2, 4997, 1003), 512, device)
    x, w, ctx = p.x_local, p.w, p.ctx
    dy = p.dy_local
    _, hist = cake_causal_conv1d_det_train_kcp_forward(x, w, ctx)
    direct = {
        coop: cake_causal_conv1d_det_train_kcp_backward(
            x, w, ctx, dy, hist, cooperative=coop
        )
        for coop in (True, False)
    }
    for coop in (True, False):
        xg = x.clone().requires_grad_(True)
        wg = w.clone().requires_grad_(True)
        y = cake_causal_conv1d_det_kcp(xg, wg, ctx, cooperative=coop)
        y.backward(dy)
        assert torch.equal(_bits(xg.grad), _bits(direct[coop][0]))
        assert torch.equal(_bits(wg.grad), _bits(direct[coop][1])), (
            f"autograd dweight differs from the direct backward (cooperative={coop})"
        )
    try:
        for coop in (True, False, True, False, True, False):
            dx, dw = cake_causal_conv1d_det_train_kcp_backward(
                x, w, ctx, dy, hist, graph="replay", cooperative=coop
            )
            assert torch.equal(dx, direct[coop][0]) and torch.equal(
                dw, direct[coop][1]
            ), f"graph replay differs from eager (cooperative={coop})"
        assert conv_kcp.LAST_CALL["bwd"]["graph_event"] == "replay"
        assert conv_kcp.GRAPH_STATS["fallback"] == 0
    finally:
        conv_kcp._GRAPHS.clear()
    for bad in (1, "split"):
        with pytest.raises(TypeError, match="cooperative"):
            cake_causal_conv1d_det_train_kcp_backward(
                x, w, ctx, dy, hist, cooperative=bad
            )
        with pytest.raises(TypeError, match="cooperative"):
            cake_causal_conv1d_det_kcp(x, w, ctx, cooperative=bad)


# ----------------------------------------------------------------------------------------------------------------
# Determinism of every output: ``REPEATS`` calls per launch form on the same inputs, the halo rows of both exchanges
# included (the forward's history rows and the backward's ``dpre`` head rows, the halo adjoints).


class _HaloRecorder:
    """Records the halo rows of every ``kcp._exchange`` call: a copy of the rows this rank sends, taken before the
    exchange (the forward's history rows, the backward's ``dpre`` head rows), and the pool it receives (``hist``,
    ``dpre_tail``; with one rank the send rows themselves)."""

    def __init__(self, monkeypatch, conv_kcp) -> None:
        self.rows: list[torch.Tensor] = []
        original = conv_kcp._exchange

        def recording(send, **kw):
            sent = send.detach().clone()
            recv, work = original(send, **kw)
            self.rows += [sent, recv]
            return recv, work

        monkeypatch.setattr(conv_kcp, "_exchange", recording)

    def take(self) -> list[torch.Tensor]:
        rows, self.rows = self.rows, []
        return rows


@pytest.mark.parametrize(
    "name,lens,cuts,C", ROWS + ROWS_8365, ids=[r[0] for r in ROWS + ROWS_8365]
)
def test_every_output_repeats_bitwise_per_launch_form(
    monkeypatch, name, lens, cuts, C
) -> None:
    """``REPEATS`` explicit forward + backward calls per launch form on the same inputs: ``y``, ``hist``, ``dx``,
    ``dweight`` and the halo rows of both exchanges repeat bit for bit; ``y``, ``hist``, ``dx`` and the halo rows are
    the same in the two forms (``dweight`` is compared within a form: each form sums its own partial rows); the
    autograd entry returns the bits of the explicit calls of its form, ``cooperative=None`` those of
    ``COOPERATIVE_DEFAULT``."""
    from flashinfer.experimental.cake.cake_causal_conv1d_det_train import (
        kcp as conv_kcp,
    )

    device = torch.device("cuda")
    p = _problem(lens, cuts, C, device)
    halo = _HaloRecorder(monkeypatch, conv_kcp)
    first = {}
    for coop in (True, False):
        for repeat in range(REPEATS):
            y, hist = cake_causal_conv1d_det_train_kcp_forward(p.x_local, p.w, p.ctx)
            dx, dw = cake_causal_conv1d_det_train_kcp_backward(
                p.x_local, p.w, p.ctx, p.dy_local, hist, cooperative=coop
            )
            torch.cuda.synchronize()
            got = {"y": y, "hist": hist, "dx": dx, "dw": dw, "halo": halo.take()}
            # one exchange per direction: (sent, received) of the forward, then of the backward
            assert len(got["halo"]) == 4, (coop, repeat, len(got["halo"]))
            if repeat == 0:
                first[coop] = got
                continue
            ref = first[coop]
            for key in ("y", "hist", "dx", "dw"):
                assert torch.equal(_bits(got[key]), _bits(ref[key])), (
                    f"cooperative={coop}: {key} of repeat {repeat} differs"
                )
            for i, (a, b) in enumerate(zip(got["halo"], ref["halo"], strict=True)):
                assert torch.equal(_bits(a), _bits(b)), (
                    f"cooperative={coop}: halo rows {i} of repeat {repeat} differ"
                )
    for key in ("y", "hist", "dx"):
        assert torch.equal(_bits(first[True][key]), _bits(first[False][key])), key
    for i, (a, b) in enumerate(
        zip(first[True]["halo"], first[False]["halo"], strict=True)
    ):
        assert torch.equal(_bits(a), _bits(b)), f"halo rows {i} differ between forms"
    for coop in (None, True, False):
        xg = p.x_local.clone().requires_grad_(True)
        wg = p.w.clone().requires_grad_(True)
        y = cake_causal_conv1d_det_kcp(xg, wg, p.ctx, cooperative=coop)
        y.backward(p.dy_local)
        ref = first[conv_kcp.COOPERATIVE_DEFAULT if coop is None else coop]
        assert torch.equal(_bits(y.detach()), _bits(ref["y"])), coop
        assert torch.equal(_bits(xg.grad), _bits(ref["dx"])), coop
        assert torch.equal(_bits(wg.grad), _bits(ref["dw"])), coop


def test_steady_state_calls_do_not_block_or_allocate() -> None:
    """Census (``tests/experimental/test_helpers/cake_host_sync.py``) of a steady round on one context: the explicit forward, the
    backward in both launch forms, the autograd entry and the three-tensor multi forward / backward.  No pageable or
    pinned host-to-device copy, no synchronizing operation (the probe queued before the round still runs when the
    host returns), no device allocation or free once the first rounds have run; bitwise the first round."""
    from flashinfer.experimental.cake.cake_causal_conv1d_det_train import (
        kcp as conv_kcp,
    )
    from tests.experimental.test_helpers.cake_host_sync import (
        allocator_counts,
        census,
        nonblocking,
        summary,
    )

    device = torch.device("cuda")
    alloc_ok = allocator_counts(device)
    p = _problem((5000, 1003), (1, 2, 4997, 1003), 512, device)
    xs, ws, dys = (p.x_local,) * 3, (p.w,) * 3, (p.dy_local,) * 3

    def rnd():
        y, hist = cake_causal_conv1d_det_train_kcp_forward(p.x_local, p.w, p.ctx)
        res = [y, hist]
        for coop in (True, False):
            res += cake_causal_conv1d_det_train_kcp_backward(
                p.x_local, p.w, p.ctx, p.dy_local, hist, cooperative=coop
            )
        xg = p.x_local.clone().requires_grad_(True)
        wg = p.w.clone().requires_grad_(True)
        yg = cake_causal_conv1d_det_kcp(xg, wg, p.ctx)
        yg.backward(p.dy_local)
        res += [yg.detach(), xg.grad, wg.grad]
        ys, hists = conv_kcp.cake_causal_conv1d_det_train_kcp_multi_forward(
            xs, ws, p.ctx
        )
        dxs, dws = conv_kcp.cake_causal_conv1d_det_train_kcp_multi_backward(
            xs, ws, p.ctx, dys, hists
        )
        return res + [*ys, *hists, *dxs, *dws]

    first = rnd()
    rnd()
    c = census(rnd, device)
    print(f"CENSUS_CONV_KCP {summary(c)}")
    assert nonblocking(c), summary(c)
    assert c["pinned"] == 0, summary(c)
    if alloc_ok:
        assert (c["device_alloc"], c["device_free"], c["alloc_retries"]) == (
            0,
            0,
            0,
        ), summary(c)
        assert c["prof_malloc"] in (None, 0), summary(c)
    for i, (a, b) in enumerate(zip(first, c["out"], strict=True)):
        assert torch.equal(_bits(a), _bits(b)), i
    if not alloc_ok:
        pytest.skip(
            "the caching allocator's statistics do not count device allocations here"
        )


# ----------------------------------------------------------------------------------------------------------------
# The backend reads no environment variable, and a copy of the package imports with flashinfer blocked.


def test_kcp_backend_reads_no_environment_variable() -> None:
    """``kcp.py`` and ``kcp_jit.py`` import no ``os`` and name no ``environ`` / ``getenv``; the private test hook
    ``_set_test_knobs`` is a plain setter (no environment gate, no ``TEST_KNOBS_ENV``)."""
    import ast
    from pathlib import Path

    from flashinfer.experimental.cake.cake_causal_conv1d_det_train import (
        kcp as conv_kcp,
    )

    package = Path(conv_kcp.__file__).resolve().parent
    banned = {"environ", "environb", "getenv", "putenv", "unsetenv"}
    for name in ("kcp.py", "kcp_jit.py"):
        tree = ast.parse((package / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            where = (name, getattr(node, "lineno", None))
            if isinstance(node, ast.Import):
                assert all(a.name.split(".")[0] != "os" for a in node.names), where
            elif isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] != "os", where
            elif isinstance(node, ast.Attribute):
                assert node.attr not in banned, where
            elif isinstance(node, ast.Name):
                assert node.id not in banned, where
    assert not hasattr(conv_kcp, "TEST_KNOBS_ENV")
    try:
        assert conv_kcp._set_test_knobs({"host_trim": 0}) == {
            "graph": "off",
            "host_trim": 0,
            "bwd_overlap": 1,
            "fwd_main": None,
            "fwd_overlap": None,
        }
        assert conv_kcp._set_test_knobs({"graph": "replay"}) == {
            "graph": "replay",
            "host_trim": 1,
            "bwd_overlap": 1,
            "fwd_main": None,
            "fwd_overlap": None,
        }
    finally:
        conv_kcp._set_test_knobs(None)
    assert conv_kcp._TEST_KNOBS == {
        "graph": "off",
        "host_trim": 1,
        "bwd_overlap": 1,
        "fwd_main": None,
        "fwd_overlap": None,
    }


_VENDORED = """
import importlib
import importlib.abc
import os
import sys
import types
import warnings

PKG_ROOT, PKG_NAME = sys.argv[1], sys.argv[2]
for p in sys.argv[3:]:
    if p not in sys.path:
        sys.path.append(p)


class _Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "flashinfer" or name.startswith("flashinfer."):
            raise ImportError(f"flashinfer is blocked in this test: {name}")
        return None


sys.meta_path.insert(0, _Block())
sys.modules["flashinfer"] = None
sys.path.insert(0, PKG_ROOT)

import torch

assert not torch.cuda.is_available()
pkg = importlib.import_module(PKG_NAME)
kcp = importlib.import_module(PKG_NAME + ".kcp")
kcp_jit = importlib.import_module(PKG_NAME + ".kcp_jit")
assert os.path.realpath(kcp.__file__).startswith(os.path.realpath(PKG_ROOT) + os.sep), kcp.__file__
for name in kcp.__all__:
    getattr(kcp, name)
assert isinstance(kcp_jit.MODULES, dict) and kcp_jit.MODULES
assert kcp.COOPERATIVE_DEFAULT is False and kcp._cooperative(None) is False
assert kcp._set_test_knobs({"host_trim": 0})["host_trim"] == 0
assert kcp._set_test_knobs(None) == {
    "graph": "off", "host_trim": 1, "bwd_overlap": 1, "fwd_main": None, "fwd_overlap": None
}

x = torch.zeros(8, 128, dtype=torch.bfloat16)
w = torch.zeros(128, 4, dtype=torch.float32)
ctx = types.SimpleNamespace(cu_seqlens_local_cpu=(0, 8))
# Host-side refusals, each before any device query, program bind or collective:
calls = [
    (ValueError, lambda: kcp.cake_causal_conv1d_det_train_kcp_forward(x, w, None, graph="eager")),
    (ValueError, lambda: kcp.cake_causal_conv1d_det_train_kcp_backward(x, w, None, x, x, graph="eager")),
    (ValueError, lambda: kcp.cake_causal_conv1d_det_train_kcp_multi_forward([x], [w], None, graph="eager")),
    (TypeError, lambda: kcp.cake_causal_conv1d_det_kcp(x, w, None, cooperative=1)),
    (TypeError, lambda: kcp.cake_causal_conv1d_det_kcp_multi([x], [w], None, cooperative="split")),
    (ValueError, lambda: kcp._kcp_forward(x, w, ctx)),  # CPU operands
]
with warnings.catch_warnings(record=True) as rec:
    warnings.simplefilter("always")
    for error, call in calls:
        try:
            call()
        except error:
            pass
        else:
            raise AssertionError("the host-side check must reject the call")
assert not rec, [f"{w.category.__name__}: {w.message}" for w in rec]
leaked = sorted(
    k for k in sys.modules if (k == "flashinfer" and sys.modules[k] is not None) or k.startswith("flashinfer.")
)
assert not leaked, leaked
print("OK")
"""


def test_vendored_kcp_imports_with_flashinfer_blocked(tmp_path) -> None:
    """A copy of the package directory under another top-level name imports its KCP module in a ``python -I``
    subprocess with ``flashinfer`` blocked (meta-path finder plus ``sys.modules["flashinfer"] = None``; no CUDA
    device): ``kcp.__all__`` resolves, the ``kcp_jit`` registry loads, ``COOPERATIVE_DEFAULT`` is ``False``, the
    private test hook needs no environment variable, six host-side refusals raise without a warning, and no
    ``flashinfer`` module is loaded."""
    import os
    import shutil
    import site
    import subprocess
    import sys
    from pathlib import Path

    import tvm_ffi

    from flashinfer.experimental.cake import cake_causal_conv1d_det_train as pkg

    name = "vendored_causal_conv1d_det_train"
    shutil.copytree(
        Path(pkg.__file__).resolve().parent,
        tmp_path / name,
        ignore=shutil.ignore_patterns("__pycache__", "aot", "csrc"),
    )
    # ``python -I`` drops PYTHONPATH and the user site: name where torch and tvm_ffi live
    paths = list(site.getsitepackages())
    for module in (torch, tvm_ffi):
        paths.append(os.path.dirname(os.path.dirname(os.path.abspath(module.__file__))))
    proc = subprocess.run(
        [sys.executable, "-I", "-c", _VENDORED, str(tmp_path), name]
        + list(dict.fromkeys(paths)),
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    assert proc.returncode == 0 and proc.stdout.strip().endswith("OK"), (
        proc.stdout,
        proc.stderr,
    )
