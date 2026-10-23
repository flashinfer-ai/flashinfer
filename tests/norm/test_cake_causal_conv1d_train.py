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

Tests for the Cake packed width-4 depthwise causal convolution + SiLU training
forward / backward (SM100 / SM103 / SM107).

Every kernel output is compared with a plain-torch reference of the same math
in FP32 with document-local shifts (``dweight`` accumulated in FP64): BF16
``y`` and ``dx`` within ``atol 1e-3 + rtol 2^-7`` (one BF16 ulp) with no
element outside, FP32 ``dweight`` within ``atol 5e-2 + rtol 1e-4``.  The
matrix covers token counts 1, 5, 63, 64, 65, 1000 and 4097 with documents of
lengths 1, 2 and 3 and empty documents before, between and after others,
channel counts 128, 136, 256, 384, 512, 768, 1024 and 2048, the ``[1, S, C]``
layout, document isolation (a change inside one document moves no output of
another), bitwise reproducibility of ``dweight``, CUDA Graph capture and
replay of changing shapes, argument validation, the workspace helpers and one
``S x C > 2^31`` case guarded by free device memory.  An architecture whose
backward route ships several programs selects one by the channel count: the
suite checks that exactly one runs per call, that every program of the route
runs for some channel count on both sides of its threshold (with a
NaN-poisoned workspace), that the workspace is sized for the largest grid of
the route, and that a routed row's ``dx`` equals the fallback program's bits.
The backward is two launches on one stream -- the selected program, then its
``reduce`` program with programmatic dependent launch -- and the suite holds
the launch order, the registry's two grid-rule kinds and FlashInfer issue
#7038's recipe: a kernel resident on another stream does not stall the
backward (``busy <= 1.5 x alone``) and leaves ``dweight`` bitwise unchanged.

This suite exercises the backend package alone: it names the package once,
through ``tests.test_helpers.cake_package``, and imports nothing else from
flashinfer, so a build that vendors the package under another import path
runs it unchanged after pointing that table entry at its package.  The
``flashinfer.cake_causal_conv1d_train`` facade and the ``flashinfer``
top-level exposure are covered by ``test_cake_causal_conv1d_train_api.py``.
"""

from __future__ import annotations

import re
import statistics
from pathlib import Path

import pytest
import torch

from tests.test_helpers.cake_package import cake_package

pkg = cake_package("cake_causal_conv1d_train")
cake_build = cake_package("cake_causal_conv1d_train", "cake_build")
loader = cake_package("cake_causal_conv1d_train", "cake_jit")
kcp_loader = cake_package("cake_causal_conv1d_train", "kcp_jit")
cake_causal_conv1d = pkg.cake_causal_conv1d
cake_causal_conv1d_train_backward = pkg.cake_causal_conv1d_train_backward
cake_causal_conv1d_train_backward_workspace = (
    pkg.cake_causal_conv1d_train_backward_workspace
)
cake_causal_conv1d_train_backward_workspace_bytes = (
    pkg.cake_causal_conv1d_train_backward_workspace_bytes
)
cake_causal_conv1d_train_forward = pkg.cake_causal_conv1d_train_forward
is_cake_causal_conv1d_train_supported = pkg.is_cake_causal_conv1d_train_supported

SUPPORTED_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
WIDTH = 4
# The contract's tolerances: y / dx within one BF16 ulp of the FP32 reference
# (no element outside), dweight within atol 5e-2 + rtol 1e-4 of the FP64 one.
Y_ATOL, Y_RTOL = 1e-3, 2**-7
DW_ATOL, DW_RTOL = 5e-2, 1e-4
# The reference's token block (its FP32 temporaries stay bounded at S x C > 2^31).
BLOCK = 8192
# Elements compared per chunk (256 MiB of FP32 per temporary).
COMPARE_CHUNK = 2**26
# The perf packing: floor(fraction x S) tokens per document, nudged off a
# multiple of 64, the rest in the fourth document.
PERF_FRACTIONS = (0.37, 0.29, 0.21)
SEED = 2033

# (S, C, documents): documents of lengths 1, 2 and 3; empty documents before,
# between and after others; C = 136 (forward: 17 channel vectors, 7 sub-tiles of 17
# threads, 9 threads idle); boundaries inside a sub-tile and at S = 64 and 65;
# the perf packing at S = 1000; 38 documents (a six-step device-side search);
# S = 5 at the backward's channel thresholds 512 and 1024 (fewer tokens than
# one eight-token stage of the big-channel backward program).
CASES = (
    (1, 128, None),
    (5, 136, (2, 0, 3)),
    (63, 384, (1, 2, 3, 57)),
    (64, 128, None),
    (65, 256, (64, 1)),
    (1000, 512, "perf"),
    (1000, 256, (27,) * 37 + (1,)),
    (4097, 1024, (0, 4097, 0)),
    (4097, 2048, (1000, 0, 3, 3094)),
    (4097, 768, (1000, 0, 3, 3094)),
    (5, 512, (2, 0, 3)),
    (5, 1024, (2, 0, 3)),
)
LARGE_CASE = (262401, 8192, "perf")
LARGE_CASE_FREE_BYTES = 40 * 2**30
GRAPH_CASES = (
    (65, 256, (64, 1)),
    (5, 136, (2, 0, 3)),
    (1000, 512, "perf"),
    (4097, 2048, (1000, 0, 3, 3094)),
    (63, 384, (1, 2, 3, 57)),
)
GRAPH_REPLAYS = 4


def _device() -> torch.device:
    return torch.device("cuda:0")


def _capability() -> tuple[int, int]:
    return torch.cuda.get_device_capability(_device())


def _supported() -> bool:
    return (
        torch.cuda.is_available()
        and _capability() in SUPPORTED_CAPABILITIES
        and is_cake_causal_conv1d_train_supported(_device())
    )


pytestmark = pytest.mark.skipif(
    not _supported(),
    reason="cake_causal_conv1d_train needs an SM100 / SM103 / SM107 device with exported routes",
)


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    # Build the local architecture's libraries that have no prebuilt file before
    # the first test, through the spec objects the package's loader itself builds
    # on first use (``cake_build.load_library``); a prebuilt or already cached
    # library loads without compiling.
    if _supported():
        arch = loader.arch_for_capability(_capability())
        for spec in loader.specs_for_arch(arch):
            spec.build_and_load()
    yield


def _case_id(case) -> str:
    S, C, docs = case
    tag = "one" if docs is None else ("perf" if docs == "perf" else f"{len(docs)}docs")
    return f"S{S}-C{C}-{tag}"


def _arch() -> str:
    return loader.arch_for_capability(_capability())


def _backward_thresholds(arch: str) -> tuple[int, ...]:
    """The channel thresholds of the backward programs ``arch`` selects by channel count (empty when the
    route ships the fallback program only)."""

    thresholds = set()
    for name in loader.route_modules(arch, "bwd"):
        rule = loader.select_rule(loader.MODULES[name])
        if rule is not None and arch in rule["channels"]:
            thresholds.add(int(rule["channels"][arch]))
    return tuple(sorted(thresholds))


def _fallback_program(arch: str) -> str:
    """The backward program of ``arch`` without a selection rule: the one every unrouted channel count runs."""

    names = [
        name
        for name in loader.route_modules(arch, "bwd")
        if loader.select_rule(loader.MODULES[name]) is None
    ]
    assert len(names) == 1, f"{arch}: fallback backward programs {names}"
    return names[0]


def _generator(*keys: int) -> torch.Generator:
    seed = SEED
    for key in keys:
        seed = (seed * 1_000_003 + int(key)) % (2**31 - 1)
    return torch.Generator(device=_device()).manual_seed(seed)


def doc_lengths(S: int, docs) -> tuple[int, ...]:
    if docs is None:
        return (S,)
    if docs == "perf":
        lens = []
        for fraction in PERF_FRACTIONS:
            n = int(fraction * S)
            if n % 64 == 0:
                n -= 1
            lens.append(n)
        lens.append(S - sum(lens))
        assert all(n > 0 for n in lens), f"S={S} is too short for the perf packing"
        return tuple(lens)
    lens = tuple(int(n) for n in docs)
    assert sum(lens) == S and all(n >= 0 for n in lens)
    return lens


def _cu_seqlens(S: int, docs) -> torch.Tensor:
    lens = doc_lengths(S, docs)
    return torch.tensor(
        [0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32, device=_device()
    )


def _doc_ids(S: int, cu: torch.Tensor) -> torch.Tensor:
    """The document of each token as the kernels find it: the largest ``d`` with ``cu[d] <= t``."""

    return torch.bucketize(torch.arange(S, device=cu.device), cu[1:].long(), right=True)


def _inputs(S: int, C: int, docs, *, key: int = 0):
    generator = _generator(S, C, key)
    x = torch.randn(
        S, C, dtype=torch.float32, device=_device(), generator=generator
    ).to(torch.bfloat16)
    # N(0, 0.25): variance 0.25, standard deviation 0.5.
    w = 0.5 * torch.randn(
        C, WIDTH, dtype=torch.float32, device=_device(), generator=generator
    )
    dy = torch.randn(
        S, C, dtype=torch.float32, device=_device(), generator=generator
    ).to(torch.bfloat16)
    cu = _cu_seqlens(S, docs)
    return x, w, cu, dy


def _shift_down(v: torch.Tensor, ids: torch.Tensor, d: int) -> torch.Tensor:
    """``out[t] = v[t - d]`` where token ``t - d`` exists and is of token ``t``'s document, else 0."""

    out = torch.zeros_like(v)
    if d == 0:
        return v.clone()
    if d < v.shape[0]:
        same = (ids[d:] == ids[:-d]).unsqueeze(1)
        out[d:] = torch.where(same, v[:-d], 0.0)
    return out


def _shift_up(v: torch.Tensor, ids: torch.Tensor, d: int) -> torch.Tensor:
    """``out[t] = v[t + d]`` where token ``t + d`` exists and is of token ``t``'s document, else 0."""

    out = torch.zeros_like(v)
    if d == 0:
        return v.clone()
    if d < v.shape[0]:
        same = (ids[:-d] == ids[d:]).unsqueeze(1)
        out[:-d] = torch.where(same, v[d:], 0.0)
    return out


def _preactivation(x: torch.Tensor, ids: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``pre`` of a window of FP32 rows ``x`` in tap order 0..3 (tap ``k`` multiplies ``x[t - 3 + k]``)."""

    pre = torch.zeros_like(x)
    for k in range(WIDTH):
        pre = pre + w[:, k] * _shift_down(x, ids, WIDTH - 1 - k)
    return pre


def reference_forward(
    x: torch.Tensor, w: torch.Tensor, cu: torch.Tensor
) -> torch.Tensor:
    S = x.shape[0]
    ids = _doc_ids(S, cu)
    y = torch.empty_like(x)
    for a in range(0, S, BLOCK):
        b = min(a + BLOCK, S)
        lo = max(a - (WIDTH - 1), 0)
        pre = _preactivation(x[lo:b].float(), ids[lo:b], w)[a - lo :]
        y[a:b] = (pre * torch.sigmoid(pre)).bfloat16()
    return y


def reference_backward(
    x: torch.Tensor, w: torch.Tensor, cu: torch.Tensor, dy: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``dx`` (BF16) and ``dweight`` accumulated in FP64 (policy A: the SiLU derivative of the BF16-rounded
    preactivation)."""

    S, C = x.shape
    ids = _doc_ids(S, cu)
    dx = torch.empty_like(x)
    dw = torch.zeros(C, WIDTH, device=x.device, dtype=torch.float64)
    for a in range(0, S, BLOCK):
        b = min(a + BLOCK, S)
        lo, hi = max(a - (WIDTH - 1), 0), min(b + (WIDTH - 1), S)
        xf, win = x[lo:hi].float(), ids[lo:hi]
        p = _preactivation(xf, win, w).bfloat16().float()
        s = torch.sigmoid(p)
        dpre = dy[lo:hi].float() * (s * (1.0 + p * (1.0 - s)))
        dxw = torch.zeros_like(dpre)
        for k in range(WIDTH):
            dxw = dxw + w[:, k] * _shift_up(dpre, win, WIDTH - 1 - k)
        dx[a:b] = dxw[a - lo : b - lo].bfloat16()
        rows = slice(a - lo, b - lo)
        for k in range(WIDTH):
            xs = _shift_down(xf, win, WIDTH - 1 - k)[rows].double()
            dw[:, k] += (dpre[rows].double() * xs).sum(dim=0)
    return dx, dw


def _assert_within(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
    label: str,
) -> None:
    """No element outside ``atol + rtol |expected|`` (an exact mismatch count of zero)."""

    assert tuple(actual.shape) == tuple(expected.shape), label
    flat_actual, flat_expected = actual.reshape(-1), expected.reshape(-1)
    count, first = 0, None
    # Chunked so the FP32 temporaries of the S x C > 2^31 case stay bounded.
    for start in range(0, flat_actual.numel(), COMPARE_CHUNK):
        a = flat_actual[start : start + COMPARE_CHUNK].float()
        b = flat_expected[start : start + COMPARE_CHUNK].float()
        assert torch.isfinite(a).all(), f"{label} has non-finite values"
        outside = (a - b).abs() > atol + rtol * b.abs()
        chunk_count = int(outside.sum().item())
        if chunk_count and first is None:
            index = int(outside.nonzero()[0].item())
            first = (start + index, a[index].item(), b[index].item())
        count += chunk_count
    if count:
        raise AssertionError(
            f"{label}: {count} of {flat_actual.numel()} elements outside atol={atol} rtol={rtol}; first at flat "
            f"index {first[0]}: got {first[1]!r}, expected {first[2]!r}"
        )


def _check(
    S: int,
    C: int,
    docs,
    *,
    x=None,
    w=None,
    cu=None,
    dy=None,
    workspace=None,
    label: str = "",
) -> None:
    if x is None:
        x, w, cu, dy = _inputs(S, C, docs)
    label = label or f"S={S} C={C} docs={docs}"
    y = cake_causal_conv1d_train_forward(x, w, cu)
    assert (
        y.dtype == torch.bfloat16
        and tuple(y.shape) == tuple(x.shape)
        and y.is_contiguous()
    )
    _assert_within(
        y, reference_forward(x, w, cu), atol=Y_ATOL, rtol=Y_RTOL, label=f"y {label}"
    )
    dx, dw = cake_causal_conv1d_train_backward(x, w, cu, dy, workspace=workspace)
    assert (
        dx.dtype == torch.bfloat16
        and tuple(dx.shape) == tuple(x.shape)
        and dx.is_contiguous()
    )
    assert (
        dw.dtype == torch.float32
        and tuple(dw.shape) == (C, WIDTH)
        and dw.is_contiguous()
    )
    dx_ref, dw_ref = reference_backward(x, w, cu, dy)
    _assert_within(dx, dx_ref, atol=Y_ATOL, rtol=Y_RTOL, label=f"dx {label}")
    _assert_within(
        dw, dw_ref.float(), atol=DW_ATOL, rtol=DW_RTOL, label=f"dweight {label}"
    )


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_forward_backward_match_reference(case) -> None:
    _check(*case)


def test_large_case_over_two_to_the_31_elements() -> None:
    free, _total = torch.cuda.mem_get_info(_device())
    if free < LARGE_CASE_FREE_BYTES:
        pytest.skip(
            f"needs {LARGE_CASE_FREE_BYTES >> 30} GiB free, {free >> 30} GiB available"
        )
    S, C, docs = LARGE_CASE
    assert S * C > 2**31
    _check(S, C, docs)
    torch.cuda.empty_cache()


@pytest.mark.parametrize(
    "case", ((65, 256, (64, 1)), (1000, 512, "perf")), ids=_case_id
)
def test_batched_layout_matches_the_packed_one(case) -> None:
    """``[1, S, C]`` inputs produce the same bits as ``[S, C]`` inputs, with ``[1, S, C]`` outputs."""

    S, C, docs = case
    x, w, cu, dy = _inputs(S, C, docs)
    y = cake_causal_conv1d_train_forward(x, w, cu)
    dx, dw = cake_causal_conv1d_train_backward(x, w, cu, dy)
    y3 = cake_causal_conv1d_train_forward(x.unsqueeze(0), w, cu)
    dx3, dw3 = cake_causal_conv1d_train_backward(x.unsqueeze(0), w, cu, dy.unsqueeze(0))
    assert tuple(y3.shape) == (1, S, C) and tuple(dx3.shape) == (1, S, C)
    assert torch.equal(y3[0], y) and torch.equal(dx3[0], dx) and torch.equal(dw3, dw)


def test_document_isolation() -> None:
    """A change inside one document moves no ``y`` or ``dx`` element of another document."""

    S, C, docs = 1000, 256, (301, 0, 257, 3, 439)
    x, w, cu, dy = _inputs(S, C, docs)
    lens = doc_lengths(S, docs)
    starts = [0, *torch.tensor(lens).cumsum(0).tolist()]
    y = cake_causal_conv1d_train_forward(x, w, cu)
    dx, _dw = cake_causal_conv1d_train_backward(x, w, cu, dy)
    # Perturb every token of document 2 (the 257-token one) in x and dy.
    a, b = starts[2], starts[3]
    x2 = x.clone()
    x2[a:b] = (x2[a:b].float() * -1.5 + 0.25).to(torch.bfloat16)
    dy2 = dy.clone()
    dy2[a:b] = (dy2[a:b].float() * 2.0).to(torch.bfloat16)
    y2 = cake_causal_conv1d_train_forward(x2, w, cu)
    dx2, _dw2 = cake_causal_conv1d_train_backward(x2, w, cu, dy2)
    assert not torch.equal(y2[a:b], y[a:b]), "the perturbed document did not change"
    outside = torch.ones(S, dtype=torch.bool, device=_device())
    outside[a:b] = False
    assert torch.equal(y2[outside], y[outside]), "y of another document changed"
    assert torch.equal(dx2[outside], dx[outside]), "dx of another document changed"
    _check(S, C, docs, x=x2, w=w, cu=cu, dy=dy2, label="perturbed documents")


@pytest.mark.parametrize(
    "case", ((1000, 512, "perf"), (4097, 2048, (1000, 0, 3, 3094))), ids=_case_id
)
def test_dweight_is_bitwise_identical_over_three_launches(case) -> None:
    S, C, docs = case
    x, w, cu, dy = _inputs(S, C, docs)
    workspace = cake_causal_conv1d_train_backward_workspace(S, C, _device())
    results = []
    for _repeat in range(3):
        dx = torch.full((S, C), float("nan"), dtype=torch.bfloat16, device=_device())
        dw = torch.full((C, WIDTH), float("nan"), dtype=torch.float32, device=_device())
        # Poison the workspace: the kernel must rewrite every partial row it reduces.
        workspace.view(torch.float32).fill_(float("nan"))
        cake_causal_conv1d_train_backward(
            x, w, cu, dy, workspace=workspace, dx=dx, dweight=dw
        )
        torch.cuda.synchronize()
        results.append((dx.clone(), dw.clone()))
    fresh_dx, fresh_dw = cake_causal_conv1d_train_backward(x, w, cu, dy)
    for dx, dw in results[1:]:
        assert torch.equal(dx, results[0][0])
        assert torch.equal(dw, results[0][1])
    assert torch.equal(fresh_dw, results[0][1]), "a fresh workspace changed dweight"
    assert torch.equal(fresh_dx, results[0][0])
    assert torch.isfinite(results[0][1]).all()


def test_backward_programs_switch_on_channel_count() -> None:
    """Exactly one backward program runs per call, every program of the route runs for some channel count,
    and each matches the reference on both sides of its channel threshold with a NaN-poisoned workspace."""

    arch = _arch()
    names = loader.route_modules(arch, "bwd")
    thresholds = _backward_thresholds(arch)
    if not thresholds:
        pytest.skip(
            f"the backward route on {arch} ships one program ({names[0]}); nothing switches"
        )
    channels = sorted({C for T in thresholds for C in (T - 8, T, T + 8)} | {8192})
    launched = set()
    for S, docs in ((5, (2, 0, 3)), (1000, "perf")):
        for C in channels:
            selected = loader.selected_modules(arch, "bwd", C=C, S=S)
            assert len(selected) == 1, f"S={S} C={C}: {selected}"
            launched.add(selected[0])
            workspace = cake_causal_conv1d_train_backward_workspace(S, C, _device())
            # Poison the whole buffer (sized for the largest grid of the route): the selected program must
            # rewrite every partial row it reduces.
            workspace.view(torch.float32).fill_(float("nan"))
            _check(
                S,
                C,
                docs,
                workspace=workspace,
                label=f"S={S} C={C} docs={docs} program={selected[0]}",
            )
    assert launched == set(names), f"never launched {set(names) - launched}"


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_routed_rows_match_fallback_dx_bitwise(case) -> None:
    """A row the channel count routes to a selected program gets the same ``dx`` bits as the fallback program
    launched on its own grid and partial rows; whether ``dweight`` agrees too (the fixed-order partition
    follows each grid) is reported, not asserted."""

    S, C, docs = case
    arch = _arch()
    fallback = _fallback_program(arch)
    selected = loader.selected_modules(arch, "bwd", C=C, S=S)[0]
    if selected == fallback:
        pytest.skip(f"S={S} C={C} runs the fallback program {fallback} on {arch}")
    device = _device()
    x, w, cu, dy = _inputs(S, C, docs)
    dx, dw = cake_causal_conv1d_train_backward(x, w, cu, dy)
    record = loader.MODULES[fallback]
    scalars = loader.plan(
        S, C, int(cu.numel()) - 1, geometry=record["grid_rule"]["geometry"]
    )
    grid_x = loader.launch_grid(record, device_index=device.index)[0]
    rows = loader.partial_rows(grid_x, scalars["n_ch_tiles"])
    dx_fallback = torch.full_like(dx, float("nan"))
    dw_fallback = torch.full_like(dw, float("nan"))
    partial = torch.full(
        (rows, C, WIDTH), float("nan"), dtype=torch.float32, device=device
    )
    values = {
        "x": x,
        "w": w,
        "cu": cu,
        "dy": dy,
        "dx": dx_fallback,
        "dw": dw_fallback,
        "partial": partial,
        **{key: int(scalars[key]) for key in loader.PLAN_SCALARS},
        "rows": rows,
    }
    loader.run_module(fallback, values, device_index=device.index)
    loader.run_module(record["reduce"], values, device_index=device.index)
    torch.cuda.synchronize(device)
    assert torch.equal(dx, dx_fallback), (
        f"dx of {selected} differs from the fallback program {fallback} at S={S} C={C}"
    )
    same_dw = torch.equal(dw, dw_fallback)
    print(
        f"dweight of {selected} vs {fallback} at S={S} C={C}: "
        + ("identical" if same_dw else "differs (each grid's fixed-order partition)")
    )


def test_preallocated_outputs_are_written_in_place() -> None:
    S, C, docs = 1000, 512, "perf"
    x, w, cu, dy = _inputs(S, C, docs)
    y_ref = cake_causal_conv1d_train_forward(x, w, cu)
    dx_ref, dw_ref = cake_causal_conv1d_train_backward(x, w, cu, dy)
    out = torch.full((S, C), float("nan"), dtype=torch.bfloat16, device=_device())
    dx = torch.full((S, C), float("nan"), dtype=torch.bfloat16, device=_device())
    dw = torch.full((C, WIDTH), float("nan"), dtype=torch.float32, device=_device())
    assert cake_causal_conv1d_train_forward(x, w, cu, out=out) is out
    got = cake_causal_conv1d_train_backward(x, w, cu, dy, dx=dx, dweight=dw)
    assert got[0] is dx and got[1] is dw
    assert (
        torch.equal(out, y_ref) and torch.equal(dx, dx_ref) and torch.equal(dw, dw_ref)
    )


def test_cuda_graph_replays_match_eager() -> None:
    """One forward + backward graph per shape, replayed four times each (twenty replays), equals eager."""

    device = _device()
    captured = []
    for S, C, docs in GRAPH_CASES:
        x, w, cu, dy = _inputs(S, C, docs, key=7)
        eager_y = cake_causal_conv1d_train_forward(x, w, cu)
        eager_dx, eager_dw = cake_causal_conv1d_train_backward(x, w, cu, dy)
        y = torch.empty_like(x)
        dx = torch.empty_like(x)
        dw = torch.empty(C, WIDTH, dtype=torch.float32, device=device)
        workspace = cake_causal_conv1d_train_backward_workspace(S, C, device)

        def step(x=x, w=w, cu=cu, dy=dy, y=y, dx=dx, dw=dw, workspace=workspace):
            cake_causal_conv1d_train_forward(x, w, cu, out=y)
            cake_causal_conv1d_train_backward(
                x, w, cu, dy, workspace=workspace, dx=dx, dweight=dw
            )

        side = torch.cuda.Stream(device=device)
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):
            step()
        torch.cuda.current_stream(device).wait_stream(side)
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=side):
            step()
        torch.cuda.synchronize(device)
        # The graph holds raw pointers to x, w, cu and dy: keep them alive until the replays, or the
        # allocator hands their memory to the next shape's tensors and the replay reads those instead.
        captured.append(
            (
                graph,
                (x, w, cu, dy),
                (y, dx, dw),
                (eager_y, eager_dx, eager_dw),
                workspace,
            )
        )
    replays = 0
    for graph, _inputs_alive, outputs, eager, workspace in captured:
        for _replay in range(GRAPH_REPLAYS):
            for tensor in outputs:
                tensor.fill_(float("nan"))
            workspace.view(torch.float32).fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize(device)
            replays += 1
            for got, expected, name in zip(
                outputs, eager, ("y", "dx", "dweight"), strict=True
            ):
                assert torch.equal(got, expected), (
                    f"{name} of replay {replays} differs from eager"
                )
    assert replays == GRAPH_REPLAYS * len(GRAPH_CASES) == 20


def test_backward_launches_the_selected_program_then_its_reduce(monkeypatch) -> None:
    """A backward call is two launches: the program ``C`` selects, then that record's ``reduce`` program with
    the partial rows of the main grid as its ``rows`` scalar and ``ceil(C / 4)`` CTAs; the forward launches
    its route in order."""

    S, C, docs = 1000, 512, "perf"
    device = _device()
    x, w, cu, dy = _inputs(S, C, docs)
    launched = []

    def record_launch(name, values, *, device_index):
        launched.append((name, dict(values), device_index))

    monkeypatch.setattr(pkg, "run_module", record_launch)
    cake_causal_conv1d_train_backward(x, w, cu, dy)
    arch = _arch()
    main = loader.selected_modules(arch, "bwd", C=C, S=S)[0]
    record = loader.MODULES[main]
    assert [name for name, _, _ in launched] == [main, record["reduce"]]
    assert {index for _, _, index in launched} == {device.index}
    values, reduce_values = launched[0][1], launched[1][1]
    assert reduce_values == values, "both launches see one argument set"
    grid_x = loader.launch_grid(record, device_index=device.index)[0]
    _, _, n_ch_tiles = loader.channel_split(C, geometry=record["grid_rule"]["geometry"])
    assert values["rows"] == loader.partial_rows(grid_x, n_ch_tiles)
    assert tuple(values["partial"].shape) == (values["rows"], C, WIDTH)
    reduce = loader.MODULES[record["reduce"]]
    assert loader.launch_grid(reduce, device_index=device.index, C=C) == (C // 4, 1, 1)
    with pytest.raises(ValueError, match="needs the channel count"):
        loader.launch_grid(reduce, device_index=device.index)
    launched.clear()
    cake_causal_conv1d_train_forward(x, w, cu)
    assert [name for name, _, _ in launched] == list(loader.route_modules(arch, "fwd"))


def test_backward_keeps_its_speed_beside_a_resident_side_kernel(
    record_property,
) -> None:
    """FlashInfer issue #7038's recipe: one 1-thread kernel resident on another stream must not stall the
    backward.

    The cooperative form waited for it (0.161 ms -> 17.85 ms on GB200); the two ordinary launches run beside it.
    Timed with CUDA events on the current stream, median of 7 after 3 warm-ups: ``busy <= 1.5 x alone``, and
    ``dweight`` is bitwise equal with and without the side kernel.  Both medians and their ratio go to the
    junit properties of the test.
    """

    device = _device()
    S, C, docs = 261393, 512, "perf"
    x, w, cu, dy = _inputs(S, C, docs)
    workspace = cake_causal_conv1d_train_backward_workspace(S, C, device)
    dx = torch.empty_like(x)
    dw = torch.empty(C, WIDTH, dtype=torch.float32, device=device)
    side = torch.cuda.Stream(device=device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    def timed(with_side_kernel: bool) -> float:
        torch.cuda.synchronize(device)
        if with_side_kernel:
            with torch.cuda.stream(side):
                torch.cuda._sleep(
                    10_000_000
                )  # one 1-thread CTA, resident for milliseconds
        torch.cuda._sleep(1_000_000)  # let it become resident before the backward
        start.record()
        cake_causal_conv1d_train_backward(
            x, w, cu, dy, workspace=workspace, dx=dx, dweight=dw
        )
        end.record()
        torch.cuda.synchronize(device)
        return start.elapsed_time(end)

    for _ in range(3):
        timed(False)
    alone = statistics.median(timed(False) for _ in range(7))
    dw_alone = dw.clone()
    busy = statistics.median(timed(True) for _ in range(7))
    print(
        f"backward alone {alone:.3f} ms, beside a resident 1-thread kernel on another "
        f"stream {busy:.3f} ms ({busy / alone:.2f}x)"
    )
    record_property("backward_alone_ms", f"{alone:.4f}")
    record_property("backward_beside_side_kernel_ms", f"{busy:.4f}")
    record_property("busy_over_alone", f"{busy / alone:.3f}")
    assert torch.equal(dw, dw_alone), "dweight changed beside the side kernel"
    assert busy <= 1.5 * alone, (
        f"the backward stalled behind a kernel resident on another stream: "
        f"{busy:.3f} ms beside it vs {alone:.3f} ms alone"
    )


def test_zero_tokens_return_empty_outputs() -> None:
    C = 256
    x = torch.empty(0, C, dtype=torch.bfloat16, device=_device())
    w = torch.zeros(C, WIDTH, dtype=torch.float32, device=_device())
    cu = torch.zeros(2, dtype=torch.int32, device=_device())
    y = cake_causal_conv1d_train_forward(x, w, cu)
    assert tuple(y.shape) == (0, C) and y.dtype == torch.bfloat16
    dx, dw = cake_causal_conv1d_train_backward(x, w, cu, x)
    assert tuple(dx.shape) == (0, C)
    assert torch.equal(dw, torch.zeros_like(dw))


def test_autograd_end_to_end() -> None:
    S, C, docs = 1000, 512, "perf"
    x0, w0, cu, dy = _inputs(S, C, docs, key=3)
    x = x0.detach().clone().requires_grad_(True)
    w = w0.detach().clone().requires_grad_(True)
    y = cake_causal_conv1d(x, w, cu)
    assert torch.equal(y.detach(), cake_causal_conv1d_train_forward(x0, w0, cu))
    y.backward(dy)
    dx_ref, dw_ref = cake_causal_conv1d_train_backward(x0, w0, cu, dy)
    assert x.grad.dtype == torch.bfloat16 and torch.equal(x.grad, dx_ref)
    assert w.grad.dtype == torch.float32 and torch.equal(w.grad, dw_ref)
    # A non-contiguous incoming gradient is materialized by the wrapper.
    x = x0.detach().clone().requires_grad_(True)
    w = w0.detach().clone().requires_grad_(True)
    cake_causal_conv1d(x, w, cu).backward(dy.t().contiguous().t())
    assert torch.equal(x.grad, dx_ref) and torch.equal(w.grad, dw_ref)


def test_changing_shapes_in_one_process_do_not_rebuild() -> None:
    arch = _arch()
    # One shape per backward program of the route: C = 128 runs the fallback program, each channel threshold
    # its own program.  The shape loop below must then load nothing.
    for C in (128, *_backward_thresholds(arch)):
        _check(64, C, None)
    loaded_before = loader.load.cache_info().misses
    for S, C, docs in (
        (1, 128, None),
        (65, 256, (64, 1)),
        (5, 136, (2, 0, 3)),
        (1000, 512, "perf"),
        (4097, 2048, (1000, 0, 3, 3094)),
        (4097, 1024, (0, 4097, 0)),
    ):
        _check(S, C, docs)
    assert loader.load.cache_info().misses == loaded_before, (
        "a changed problem size rebuilt a module"
    )


def test_workspace_bytes_and_layout() -> None:
    device = _device()
    arch = _arch()
    # The capacity owner: the backward program with the largest grid of the route sizes the public buffer.
    record = loader.workspace_record(arch)
    owners = loader.workspace_owners(arch)
    assert record in owners
    assert record is loader.MODULES[_fallback_program(arch)], (
        "the public workspace size is the fallback program's, unchanged by the"
        " programs a channel count selects"
    )
    assert record["grid_rule"]["ctas_per_sm"] == max(
        owner["grid_rule"]["ctas_per_sm"] for owner in owners
    )
    geometry = record["grid_rule"]["geometry"]
    assert all(owner["grid_rule"]["geometry"] == geometry for owner in owners)
    grid = loader.launch_grid(record, device_index=device.index)[0]
    assert grid == record["grid_rule"]["ctas_per_sm"] * loader.sm_count(device.index)
    for C in (128, 136, 8192):
        small = cake_causal_conv1d_train_backward_workspace_bytes(1, C, device)
        large = cake_causal_conv1d_train_backward_workspace_bytes(262401, C, device)
        assert small == large > 0, "the workspace is independent of S"
        layout = loader.workspace_layout(record, C, device_index=device.index)
        _, _, n_ch_tiles = loader.channel_split(C, geometry=geometry)
        assert layout["grid"] == grid
        assert (
            layout["rows"]
            == loader.partial_rows(grid, n_ch_tiles)
            == -(-grid // n_ch_tiles)
        )
        assert small == layout["total_bytes"] == 4 * WIDTH * C * layout["rows"]
        assert small == max(
            loader.workspace_layout(owner, C, device_index=device.index)["total_bytes"]
            for owner in owners
        )
    # The program a channel count selects writes the leading rows of its own grid inside that buffer, never
    # more than the capacity owner's rows (both sides of every threshold and the widest perf row).
    for T in _backward_thresholds(arch):
        for C in (T - 8, T, 8192):
            selected = loader.MODULES[loader.selected_modules(arch, "bwd", C=C)[0]]
            _, _, n_ch_tiles = loader.channel_split(
                C, geometry=selected["grid_rule"]["geometry"]
            )
            rows = loader.partial_rows(
                loader.launch_grid(selected, device_index=device.index)[0], n_ch_tiles
            )
            capacity = loader.workspace_layout(record, C, device_index=device.index)
            assert rows <= capacity["rows"], (C, selected["library"])
            assert (
                loader.partial_bytes(selected, C, rows)
                <= cake_causal_conv1d_train_backward_workspace_bytes(1, C, device)
                == capacity["total_bytes"]
            )
    # One channel tile holds up to threads x vec channels (a partial row per CTA): 1024 for the backward's eight
    # channels per thread; 8192 channels are eight tiles; C = 136 is 17 channel vectors (seven sub-tiles of 17).
    vec, threads = geometry["vec"], geometry["threads"]
    assert loader.channel_split(threads * vec, geometry=geometry) == (threads, 1, 1)
    assert loader.channel_split(8192, geometry=geometry) == (
        threads,
        1,
        8192 // (threads * vec),
    )
    assert loader.channel_split(136, geometry=geometry) == (
        136 // vec,
        threads // (136 // vec),
        1,
    )
    workspace = cake_causal_conv1d_train_backward_workspace(65536, 512, device)
    assert (
        workspace.dtype == torch.uint8 and workspace.numel() == 4 * WIDTH * 512 * grid
    )
    with pytest.raises(ValueError):
        cake_causal_conv1d_train_backward_workspace_bytes(-1, 512, device)
    with pytest.raises(ValueError):
        cake_causal_conv1d_train_backward_workspace_bytes(1, 130, device)


def test_argument_validation() -> None:
    S, C = 64, 256
    x, w, cu, dy = _inputs(S, C, None)
    with pytest.raises(ValueError, match="multiple of 8"):
        cake_causal_conv1d_train_forward(
            x[:, :130].contiguous(), w[:130].contiguous(), cu
        )
    with pytest.raises(ValueError, match="width-4"):
        cake_causal_conv1d_train_forward(x, w[:, :3].contiguous(), cu)
    with pytest.raises(ValueError, match="width-4"):
        cake_causal_conv1d_train_forward(
            x, torch.cat([w, w[:, :1]], dim=1).contiguous(), cu
        )
    with pytest.raises(ValueError, match="one row per channel"):
        cake_causal_conv1d_train_forward(x, w[: C // 2].contiguous(), cu)
    with pytest.raises(ValueError, match="torch.int32"):
        cake_causal_conv1d_train_forward(x, w, cu.long())
    with pytest.raises(ValueError, match="cu_seqlens"):
        cake_causal_conv1d_train_forward(x, w, cu[:1])
    with pytest.raises(ValueError, match="shape"):
        cake_causal_conv1d_train_backward(x, w, cu, dy[: S - 1])
    with pytest.raises(ValueError, match="contiguous"):
        cake_causal_conv1d_train_forward(x.t().contiguous().t(), w, cu)
    with pytest.raises(ValueError, match="torch.bfloat16"):
        cake_causal_conv1d_train_forward(x.float(), w, cu)
    with pytest.raises(ValueError, match="torch.float32"):
        cake_causal_conv1d_train_forward(x, w.to(torch.bfloat16), cu)
    with pytest.raises(ValueError, match="shape"):
        cake_causal_conv1d_train_forward(
            x, w, cu, out=torch.empty(S + 1, C, dtype=torch.bfloat16, device=_device())
        )
    with pytest.raises(ValueError, match="contiguous"):
        cake_causal_conv1d_train_backward(
            x,
            w,
            cu,
            dy,
            dx=torch.empty(C, S, dtype=torch.bfloat16, device=_device()).t(),
        )
    with pytest.raises(ValueError, match="workspace holds"):
        cake_causal_conv1d_train_backward(
            x, w, cu, dy, workspace=torch.empty(16, dtype=torch.uint8, device=_device())
        )
    with pytest.raises(ValueError, match="CUDA"):
        cake_causal_conv1d_train_forward(x.cpu(), w.cpu(), cu.cpu())
    with pytest.raises(TypeError):
        cake_causal_conv1d_train_forward(x, w, [0, S])


def test_registry_records_reference_the_build_manifest() -> None:
    """Every ``MODULES`` record names its ``build.json`` library and nothing to compile.

    The manifest (schema ``cake-build/1``) carries exactly one entry per (library,
    architecture) the registry references, each with the kernel and binding
    sources under ``csrc/`` of this package, their digests and a closure digest;
    the registry itself records the host plan only.
    """

    manifest = cake_build.manifest()
    assert manifest["schema"] == "cake-build/1"
    assert manifest["package"] == pkg.__name__
    libraries = cake_build.libraries_by_name()
    package = Path(cake_build.__file__).parent
    referenced = set()
    routed = {
        name
        for route in loader.ROUTES.values()
        for names in route.values()
        for name in names
    }
    # The routes list the programs a call selects; the reduce programs hang off the backward records.
    reduces = {
        loader.MODULES[name]["reduce"]
        for name in routed
        if loader.MODULES[name]["role"] == "bwd"
    }
    assert set(loader.MODULES) == routed | reduces and not routed & reduces
    for name, record in loader.MODULES.items():
        assert record["library"] == record["kernel"] == name
        assert record["kernel_symbol"] == f"kernel_{name}"
        assert not {"sources", "source_sha256", "compile_flags", "cache_names"} & set(
            record
        ), name
        assert set(record["arches"]) <= set(manifest["arch_flags"]), name
        assert record["role"] in loader.ROLES, name
        grid = record["grid_rule"]
        assert grid["kind"] in loader.GRID_KINDS, name
        # No launch is cooperative (FlashInfer #7038): a main program sizes its grid by occupancy, a reduce
        # program by channels, and both are ordinary launches.
        assert grid["cooperative"] is False, name
        assert record["launch"]["cooperative"] is False, name
        if record["role"] == "reduce":
            assert grid["kind"] == "channels_per_cta", name
            assert grid["channels_per_cta"] * 32 == record["launch"]["block"][0], (
                name,
                "one warp per channel",
            )
            assert record["launch"]["use_pdl"] is True, name
            assert record["workspace_rule"] is None, name
            assert not {"select_rule", "reduce"} & set(record), name
            assert set(record["logical"].values()) == {"partial", "dw", "C", "rows"}
        else:
            # The forward keeps the co-resident grid it always launched normally; the backward programs
            # of the split are sized by occupancy.
            assert (
                grid["kind"]
                == {"fwd": "co_resident", "bwd": "occupancy"}[record["role"]]
            ), name
            assert grid["ctas_per_sm"] >= 1
            assert set(grid["geometry"]) == {"threads", "vec"}
            assert grid["geometry"]["threads"] == record["launch"]["block"][0]
            assert loader.CHANNEL_MULTIPLE % grid["geometry"]["vec"] == 0
            assert record["launch"]["use_pdl"] is False, name
        if record["role"] == "bwd":
            reduce = loader.MODULES[record["reduce"]]
            assert reduce["role"] == "reduce", name
            assert set(record["arches"]) <= set(reduce["arches"]), name
        else:
            assert "reduce" not in record, name
        plan = {key for kind, key in record["arg_plan"] if kind != "grid"}
        assert plan == set(record["logical"])
        assert set(record["logical"].values()) <= set(loader.LOGICAL_TENSORS) | set(
            loader.LOGICAL_SCALARS
        )
        # The selection vocabulary: forward records carry no rule; a backward rule is
        # ``channels_at_least`` with one positive multiple-of-8 threshold per architecture it names.
        rule = loader.select_rule(record)
        if record["role"] == "fwd":
            assert rule is None, name
        if rule is not None:
            assert record["role"] == "bwd", name
            assert set(rule) == {"kind", "channels"}, name
            assert rule["kind"] in loader.SELECT_KINDS, name
            assert set(rule["channels"]) <= set(record["arches"]), name
            for threshold in rule["channels"].values():
                assert threshold > 0 and threshold % loader.CHANNEL_MULTIPLE == 0, name
        for arch in record["arches"]:
            library = loader.library_name(name, arch)
            referenced.add(library)
            entry = libraries[library]
            assert entry["arch"] == arch
            assert entry["entry"] == "run"
            assert entry["sources"] == [
                f"csrc/{record['library']}_kernel.cu",
                f"csrc/{record['library']}_binding.cu",
            ]
            assert set(entry["sha256"]) == set(entry["sources"])
            for source in entry["sources"]:
                assert (package / source).is_file(), source
            assert re.fullmatch(r"[0-9a-f]{64}", entry["closure_sha256"])
            assert "--use_fast_math" not in entry["nvcc_flags"]
            assert "--prec-div=true" in entry["nvcc_flags"]
    # The KCP registry (kcp_jit.MODULES) adds its programs to the same build.json.
    for name, record in kcp_loader.MODULES.items():
        assert record["library"] == record["kernel"] == name
        for arch in record["arches"]:
            library = kcp_loader.library_name(name, arch)
            referenced.add(library)
            entry = libraries[library]
            assert entry["arch"] == arch
            assert entry["sources"] == [
                f"csrc/{name}_kernel.cu",
                f"csrc/{name}_binding.cu",
            ]
            assert "--use_fast_math" not in entry["nvcc_flags"]
            assert "--prec-div=true" in entry["nvcc_flags"]
    assert referenced == set(libraries)
    assert (package / "csrc" / "cake_ffi_checks.h").is_file()
    backward = loader.workspace_record("sm_100a")
    assert backward["workspace_rule"] == {"kind": "partial_rows", "taps": WIDTH}
    assert backward["launch"]["cooperative"] is False
    # Every backward route: exactly one fallback record, every routed record's rule names the
    # architecture with its own threshold (distinct thresholds), one program selected per channel
    # count -- the highest threshold at most C, the fallback below the lowest -- and the forward route
    # selects nothing.
    for arch in loader.ARCHES:
        names = loader.route_modules(arch, "bwd")
        rules = {name: loader.select_rule(loader.MODULES[name]) for name in names}
        fallbacks = [name for name, rule in rules.items() if rule is None]
        assert len(fallbacks) == 1, (arch, fallbacks)
        positives = sorted(
            (int(rule["channels"][arch]), name)
            for name, rule in rules.items()
            if rule is not None
        )
        assert all(
            arch in rules[name]["channels"] for name in names if rules[name] is not None
        ), (
            arch,
            names,
        )
        thresholds = [threshold for threshold, _ in positives]
        assert thresholds == sorted(set(thresholds)), (arch, positives)
        assert len(positives) <= 3, (
            arch,
            positives,
        )  # sm_107a routes three programs beside its fallback
        previous = fallbacks[0]
        for threshold, name in positives:
            assert loader.selected_modules(arch, "bwd", C=threshold) == (name,)
            assert loader.selected_modules(arch, "bwd", C=threshold - 8) == (previous,)
            previous = name
        assert loader.selected_modules(arch, "bwd", C=8192) == (previous,)
        for C in (8, 128, 512, 1024, 8192):
            selected = loader.selected_modules(arch, "bwd", C=C, S=1)
            assert len(selected) == 1 and selected[0] in names, (arch, C, selected)
        assert loader.selected_modules(arch, "fwd", C=512) == loader.route_modules(
            arch, "fwd"
        )
        owners = loader.workspace_owners(arch)
        assert [owner["library"] for owner in owners] == list(names)
        for owner in owners:
            assert owner["role"] == "bwd"
            assert owner["workspace_rule"] == {"kind": "partial_rows", "taps": WIDTH}
            assert owner["launch"]["cooperative"] is False
            assert arch in loader.MODULES[owner["reduce"]]["arches"], (
                arch,
                owner["library"],
            )
        assert loader.workspace_record(arch) in owners
