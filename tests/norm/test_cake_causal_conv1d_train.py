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
with document-local shifts: BF16 ``y`` and ``dx`` within ``atol 1e-3 + rtol
2^-7`` (one BF16 ulp) with no element outside; the FP32 gradients within ``atol
5e-2 + rtol 1e-4`` under the default SiLU' policy (the reference takes the SiLU
derivative at the BF16-rounded FP32 preactivation, like the kernel, and sums in
FP64) and within ``atol 1e-3 + rtol 2e-6`` under ``silu_grad_preactivation=
"fp32"`` (the reference is FP64 throughout); a BF16 gradient out is held to the
same bound plus its own rounding (``|bf16(a) - b| <= 2^-8 |b| + (1 + 2^-8)
|a - b|``: ``atol x (1 + 2^-8)``, ``rtol 2^-7``).  The matrix covers token
counts 1, 5, 63, 64, 65, 1000 and 4097 with documents of lengths 1, 2 and 3
and empty documents before, between and after others, channel counts 128,
136, 256, 384, 512, 768, 1024, 1280, 1344, 1536, 1600, 2048 and 2560 (the
round-6 tile rule splits 1280 as five 32-vector tiles of four token sub-tiles,
1536 and 2560 as 64-vector tiles of two; 1344 and 1600, with no 32-multiple
divisor, take the forward's and the backward's own fallback tiles),
the ``[1, S, C]`` layout, document isolation, bitwise reproducibility of the
gradients, CUDA Graph capture and replay of changing shapes, argument
validation, the workspace helpers and one ``S x C > 2^31`` case guarded by free
device memory.  Beside the default routes the suite covers the six round-6
routes -- BF16 taps and / or a BF16 ``dweight`` (``*_dt``), an FP32 or BF16
bias with its gradient (``*_bias``), SiLU' at the unrounded FP32 preactivation
(``bwd_fp32g``, with a perf-size rel-L2 gate against an FP64 reference) and
their combination -- each with a NaN-poisoned workspace, run-to-run bitwise
gradients and graph replay; a route an architecture has not rendered yet
skips its tests, and the registry test fails until every route's libraries
are in ``build.json``.  An architecture whose route ships several programs
(every backward route; the forward routes on sm_107a) selects one by the
channel count: the suite checks that exactly one runs per call, that every
program of the route runs for some channel count on both sides of its
threshold, that the workspace is sized for the largest partial buffer over
every route, and that a routed row's ``dx`` (forward: ``y``) equals the
fallback program's bits.  The backward is two launches on one stream -- the selected
program, then its ``reduce`` program with programmatic dependent launch -- and
the suite holds the launch order, the registry's grid-rule kinds and
FlashInfer issue #7038's recipe: a kernel resident on another stream does not
stall the backward (``busy <= 2 x alone``) and leaves ``dweight`` bitwise
unchanged.  The forward takes its token chunks from int32 counters in a
forward workspace that every launch leaves zero: the suite checks its size
rule, one caller-owned buffer over repeated calls of every forward route,
two streams with two buffers running at once, CUDA Graph capture and replay
with and without a caller-owned buffer, and the package's per-(device,
stream) buffer of the calls that pass none.

This suite exercises the backend package alone: it names the package once,
through ``tests.test_helpers.cake_package``, and imports nothing else from
flashinfer, so a build that vendors the package under another import path
runs it unchanged after pointing that table entry at its package.  The
``flashinfer.cake_causal_conv1d_train`` facade and the ``flashinfer``
top-level exposure are covered by ``test_cake_causal_conv1d_train_api.py``.
"""

from __future__ import annotations

import itertools
import re
import statistics
from fractions import Fraction
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
cake_causal_conv1d_train_forward_workspace = (
    pkg.cake_causal_conv1d_train_forward_workspace
)
cake_causal_conv1d_train_forward_workspace_bytes = (
    pkg.cake_causal_conv1d_train_forward_workspace_bytes
)
is_cake_causal_conv1d_train_supported = pkg.is_cake_causal_conv1d_train_supported

SUPPORTED_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
WIDTH = 4
F32, BF16 = torch.float32, torch.bfloat16
# The contract's tolerances: y / dx within one BF16 ulp of the reference (no
# element outside); the FP32 gradients within atol 5e-2 + rtol 1e-4 of the FP64
# sum under the default SiLU' policy and within atol 1e-3 + rtol 2e-6 of the FP64
# reference under silu_grad_preactivation="fp32".
Y_ATOL, Y_RTOL = 1e-3, 2**-7
DW_ATOL, DW_RTOL = 5e-2, 1e-4
DW_FP32G_ATOL, DW_FP32G_RTOL = 1e-3, 2e-6
# The perf-size precision gate of the fp32 policy: rel-L2 of dweight against the
# FP64 reference over the C x 4 entries at S = 250340, C 512 and 1280 (CONTRACT
# §1.5 / §4: design target 2e-7, the owner's kernel 1.5e-7).  THE PARENT SETS
# THE MEASURED VALUE HERE once the fp32 programs are rendered and measured; the
# measured rel-L2 is printed and recorded by the test for the ledger.
FP32G_REL_L2_GATE = 2e-7
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
# one eight-token stage of the big-channel backward program) and at the new
# tile geometries; C = 1280 (160 vectors: five 32-vector tiles, four token
# sub-tiles each), 1536 (192: three 64-vector tiles of two sub-tiles), 2560
# (320: five 64-vector tiles of two), and on the fallbacks (no 32-multiple
# divisor), forward | backward, 1344 (168: four whole 42-vector tiles of three
# sub-tiles | three 64-vector tiles of two, the last 40 vectors) and 1600 (200:
# five whole 40-vector tiles of three | seven 32-vector tiles of four, the last
# eight vectors).
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
    (4097, 1280, (1000, 0, 3, 3094)),
    (5, 1280, (2, 0, 3)),
    (5, 1536, (2, 0, 3)),
    (4097, 2560, (1000, 0, 3, 3094)),
    (4097, 1344, (1000, 0, 3, 3094)),
    (1000, 1600, "perf"),
)
# A short list per route (every route x every case would not add coverage of
# the route-specific code, which is the tap / bias load form and the store
# form): the two new multi-sub-tile geometries, a whole-row tile and the
# S = 5 row below one stage.
ROUTE_CASES = (
    (1000, 512, "perf"),
    (4097, 1280, (1000, 0, 3, 3094)),
    (5, 1536, (2, 0, 3)),
    (65, 256, (64, 1)),
)
LARGE_CASE = (262401, 8192, "perf")
LARGE_CASE_FREE_BYTES = 40 * 2**30
# S 250340 x C 512 and 1280 (CONTRACT §4): the perf-size rows of the fp32 gate.
PRECISION_CASES = ((250340, 512, "perf"), (250340, 1280, "perf"))
GRAPH_CASES = (
    (65, 256, (64, 1)),
    (5, 136, (2, 0, 3)),
    (1000, 512, "perf"),
    (4097, 2048, (1000, 0, 3, 3094)),
    (63, 384, (1, 2, 3, 57)),
    (4097, 1280, (1000, 0, 3, 3094)),
    (5, 1536, (2, 0, 3)),
    (4097, 2560, (1000, 0, 3, 3094)),
)
GRAPH_REPLAYS = 4
# The literal channel-split tables of CONTRACT §3 at 128 threads x 8 channels:
# C -> (ct_vec, n_sub, n_ch_tiles) under the forward's and the backward's rule
# (D1 wherever cv <= 128 or a multiple of 32 divides cv, the role's fallback at
# C 1032, 1152, 1160, 1344, 1408, 1600, 1920, 2176, 2688 and 3200) and under the
# legacy rule the KCP path keeps.
CHANNEL_SPLIT_TABLE = {
    128: ((16, 8, 1), (16, 8, 1), (16, 8, 1)),
    256: ((32, 4, 1), (32, 4, 1), (32, 4, 1)),
    384: ((48, 2, 1), (48, 2, 1), (48, 2, 1)),
    512: ((64, 2, 1), (64, 2, 1), (64, 2, 1)),
    768: ((96, 1, 1), (96, 1, 1), (96, 1, 1)),
    1024: ((128, 1, 1), (128, 1, 1), (128, 1, 1)),
    1032: ((32, 4, 5), (32, 4, 5), (128, 1, 2)),
    1152: ((32, 4, 5), (32, 4, 5), (128, 1, 2)),
    1160: ((32, 4, 5), (32, 4, 5), (128, 1, 2)),
    1280: ((32, 4, 5), (32, 4, 5), (128, 1, 2)),
    1344: ((42, 3, 4), (64, 2, 3), (128, 1, 2)),
    1408: ((64, 2, 3), (64, 2, 3), (128, 1, 2)),
    1536: ((64, 2, 3), (64, 2, 3), (128, 1, 2)),
    1600: ((40, 3, 5), (32, 4, 7), (128, 1, 2)),
    1920: ((128, 1, 2), (128, 1, 2), (128, 1, 2)),
    2048: ((128, 1, 2), (128, 1, 2), (128, 1, 2)),
    2176: ((32, 4, 9), (32, 4, 9), (128, 1, 3)),
    2560: ((64, 2, 5), (64, 2, 5), (128, 1, 3)),
    2688: ((42, 3, 8), (32, 4, 11), (128, 1, 3)),
    3200: ((32, 4, 13), (32, 4, 13), (128, 1, 4)),
    4096: ((128, 1, 4), (128, 1, 4), (128, 1, 4)),
    8192: ((128, 1, 8), (128, 1, 8), (128, 1, 8)),
}
# The call settings that resolve to each backward route (the forward route
# follows: fwd, fwd_dt, fwd_bias, fwd, fwd_bias).
ROUTE_SETTINGS = {
    "bwd": dict(weight_dtype=F32, bias_dtype=None, policy="bf16"),
    "bwd_dt": dict(weight_dtype=BF16, bias_dtype=None, policy="bf16"),
    "bwd_bias": dict(weight_dtype=F32, bias_dtype=F32, policy="bf16"),
    "bwd_fp32g": dict(weight_dtype=F32, bias_dtype=None, policy="fp32"),
    "bwd_bias_fp32g": dict(weight_dtype=F32, bias_dtype=F32, policy="fp32"),
}
NEW_BWD_KINDS = ("bwd_dt", "bwd_bias", "bwd_fp32g", "bwd_bias_fp32g")
# The forward route each backward route's inputs select.
FORWARD_OF = {
    "bwd": "fwd",
    "bwd_dt": "fwd_dt",
    "bwd_bias": "fwd_bias",
    "bwd_fp32g": "fwd",
    "bwd_bias_fp32g": "fwd_bias",
}


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


def _dtype_id(dtype) -> str:
    return {None: "none", F32: "f32", BF16: "bf16"}[dtype]


def _arch() -> str:
    return loader.arch_for_capability(_capability())


def _route_pending(arch: str, kind: str) -> str | None:
    """Why ``kind`` cannot run on ``arch`` yet (``None`` when it can): the route is not in the registry, or a
    record of it names a library ``build.json`` has no entry for (the exporter has not rendered it)."""

    if kind not in loader.ROUTES.get(arch, {}):
        return f"route {kind!r} is not exported for {arch}"
    libraries = cake_build.libraries_by_name()
    for name in loader.route_modules(arch, kind):
        library = loader.library_name(name, arch)
        if library not in libraries:
            return f"route {kind!r}: {library} is not in build.json yet (awaiting the exporter's render)"
    return None


def _require_route(*kinds: str) -> None:
    """Skip unless every route of ``kinds`` (a backward route brings its forward route along) can run."""

    for kind in kinds:
        for needed in (kind, FORWARD_OF.get(kind, kind)):
            pending = _route_pending(_arch(), needed)
            if pending:
                pytest.skip(pending)


def _backward_thresholds(arch: str, kind: str = "bwd") -> tuple[int, ...]:
    """The channel thresholds of the programs route ``kind`` selects by channel count on ``arch`` (empty when the
    route ships the fallback program only)."""

    thresholds = set()
    for name in loader.route_modules(arch, kind):
        rule = loader.select_rule(loader.MODULES[name])
        if rule is not None and arch in rule["channels"]:
            thresholds.add(int(rule["channels"][arch]))
    return tuple(sorted(thresholds))


def _token_floors(arch: str, kind: str = "bwd") -> dict[str, tuple[int, int]]:
    """The routed programs of route ``kind`` on ``arch`` whose rule carries a token floor: name -> (channel
    threshold, token floor)."""

    floors = {}
    for name in loader.route_modules(arch, kind):
        rule = loader.select_rule(loader.MODULES[name])
        if rule is not None and arch in rule.get("tokens", {}):
            floors[name] = (int(rule["channels"][arch]), int(rule["tokens"][arch]))
    return floors


def _fallback_program(arch: str, kind: str = "bwd") -> str:
    """The program of route ``kind`` without a selection rule: the one every unrouted channel count runs."""

    names = [
        name
        for name in loader.route_modules(arch, kind)
        if loader.select_rule(loader.MODULES[name]) is None
    ]
    assert len(names) == 1, f"{arch} {kind}: fallback programs {names}"
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


def _inputs(S: int, C: int, docs, *, key: int = 0, weight_dtype=F32):
    """``(x, w, cu, dy)``: BF16 ``x`` / ``dy`` of N(0, 1), taps of N(0, 0.25) in ``weight_dtype``."""

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
    return x, w.to(weight_dtype), cu, dy


def _bias_input(C: int, dtype, *, key: int = 0) -> torch.Tensor:
    """A per-channel bias of N(0, 0.25) in ``dtype`` (its own stream, so the taps stay the same with or without
    a bias)."""

    generator = _generator(C, 97, key)
    return (
        0.5 * torch.randn(C, dtype=torch.float32, device=_device(), generator=generator)
    ).to(dtype)


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


def _preactivation(
    x: torch.Tensor, ids: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None
) -> torch.Tensor:
    """``pre`` of a window of rows ``x`` in tap order 0..3 (tap ``k`` multiplies ``x[t - 3 + k]``), in ``x``'s
    dtype (FP32: the kernels' mul / add chain, bit-exact; FP64: the fp32 policy's reference), the bias added
    last."""

    pre = torch.zeros_like(x)
    for k in range(WIDTH):
        pre = pre + w[:, k].to(x.dtype) * _shift_down(x, ids, WIDTH - 1 - k)
    if b is not None:
        pre = pre + b.to(x.dtype)
    return pre


def reference_forward(
    x: torch.Tensor, w: torch.Tensor, cu: torch.Tensor, b: torch.Tensor | None = None
) -> torch.Tensor:
    S = x.shape[0]
    ids = _doc_ids(S, cu)
    y = torch.empty_like(x)
    for a in range(0, S, BLOCK):
        bb = min(a + BLOCK, S)
        lo = max(a - (WIDTH - 1), 0)
        pre = _preactivation(x[lo:bb].float(), ids[lo:bb], w, b)[a - lo :]
        y[a:bb] = (pre * torch.sigmoid(pre)).bfloat16()
    return y


def reference_backward(
    x: torch.Tensor,
    w: torch.Tensor,
    cu: torch.Tensor,
    dy: torch.Tensor,
    *,
    b: torch.Tensor | None = None,
    policy: str = "bf16",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """``dx`` (BF16), ``dweight`` and ``dbias`` (FP64 sums; ``dbias`` ``None`` without a bias).

    ``policy="bf16"``: the SiLU derivative at the BF16-rounded FP32 preactivation
    (the kernel's own FP32 ``dpre``; the chain is bit-exact, so the only
    difference to the kernel is the summation order).  ``policy="fp32"``: the
    preactivation, SiLU derivative, ``dpre`` and the ``dx`` sum in FP64, then one
    rounding (never the kernel's FMA order: the one-BF16-ulp ``dx`` bound holds
    against any FP32-or-better evaluation).
    """

    S, C = x.shape
    ids = _doc_ids(S, cu)
    dx = torch.empty_like(x)
    dw = torch.zeros(C, WIDTH, device=x.device, dtype=torch.float64)
    db = None if b is None else torch.zeros(C, device=x.device, dtype=torch.float64)
    for a in range(0, S, BLOCK):
        bb = min(a + BLOCK, S)
        lo, hi = max(a - (WIDTH - 1), 0), min(bb + (WIDTH - 1), S)
        win = ids[lo:hi]
        if policy == "fp32":
            xf = x[lo:hi].double()
            p = _preactivation(xf, win, w, b)
        else:
            xf = x[lo:hi].float()
            p = _preactivation(xf, win, w, b).bfloat16().float()
        s = torch.sigmoid(p)
        dpre = dy[lo:hi].to(xf.dtype) * (s * (1.0 + p * (1.0 - s)))
        dxw = torch.zeros_like(dpre)
        for k in range(WIDTH):
            dxw = dxw + w[:, k].to(xf.dtype) * _shift_up(dpre, win, WIDTH - 1 - k)
        dx[a:bb] = dxw[a - lo : bb - lo].bfloat16()
        rows = slice(a - lo, bb - lo)
        for k in range(WIDTH):
            xs = _shift_down(xf, win, WIDTH - 1 - k)[rows].double()
            dw[:, k] += (dpre[rows].double() * xs).sum(dim=0)
        if db is not None:
            db += dpre[rows].double().sum(dim=0)
    return dx, dw, db


def _grad_tolerance(policy: str, dtype: torch.dtype) -> tuple[float, float]:
    """``(atol, rtol)`` of a gradient out tensor: the policy's FP32 bound, widened for a BF16 out by its own
    rounding (``|bf16(a) - b| <= 2^-8 |b| + (1 + 2^-8) |a - b|``)."""

    atol, rtol = (
        (DW_ATOL, DW_RTOL) if policy == "bf16" else (DW_FP32G_ATOL, DW_FP32G_RTOL)
    )
    if dtype == torch.bfloat16:
        return atol * (1 + 2**-8), 2**-7
    return atol, rtol


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
    b=None,
    policy: str = "bf16",
    dweight_dtype=None,
    dbias_dtype=None,
    workspace=None,
    label: str = "",
) -> None:
    """Forward and backward of one problem against the reference; ``b`` adds the bias path, ``policy`` the
    SiLU' policy, ``dweight_dtype`` / ``dbias_dtype`` preallocate the gradient outs in that dtype."""

    if x is None:
        x, w, cu, dy = _inputs(S, C, docs)
    label = (
        label
        or f"S={S} C={C} docs={docs} w={w.dtype} b={None if b is None else b.dtype} {policy}"
    )
    y = cake_causal_conv1d_train_forward(x, w, cu, bias=b)
    assert (
        y.dtype == torch.bfloat16
        and tuple(y.shape) == tuple(x.shape)
        and y.is_contiguous()
    )
    _assert_within(
        y, reference_forward(x, w, cu, b), atol=Y_ATOL, rtol=Y_RTOL, label=f"y {label}"
    )
    dw_out = (
        None
        if dweight_dtype is None
        else torch.empty(C, WIDTH, dtype=dweight_dtype, device=_device())
    )
    db_out = (
        None
        if dbias_dtype is None
        else torch.empty(C, dtype=dbias_dtype, device=_device())
    )
    grads = cake_causal_conv1d_train_backward(
        x,
        w,
        cu,
        dy,
        bias=b,
        silu_grad_preactivation=policy,
        workspace=workspace,
        dweight=dw_out,
        dbias=db_out,
    )
    assert len(grads) == (2 if b is None else 3), label
    dx, dw = grads[0], grads[1]
    db = grads[2] if b is not None else None
    assert (
        dx.dtype == torch.bfloat16
        and tuple(dx.shape) == tuple(x.shape)
        and dx.is_contiguous()
    )
    assert dw.dtype == (dweight_dtype or w.dtype), label
    assert tuple(dw.shape) == (C, WIDTH) and dw.is_contiguous()
    if dw_out is not None:
        assert dw is dw_out
    dx_ref, dw_ref, db_ref = reference_backward(x, w, cu, dy, b=b, policy=policy)
    _assert_within(dx, dx_ref, atol=Y_ATOL, rtol=Y_RTOL, label=f"dx {label}")
    atol, rtol = _grad_tolerance(policy, dw.dtype)
    _assert_within(
        dw.float(), dw_ref.float(), atol=atol, rtol=rtol, label=f"dweight {label}"
    )
    if b is not None:
        assert db.dtype == (dbias_dtype or b.dtype), label
        assert tuple(db.shape) == (C,) and db.is_contiguous()
        if db_out is not None:
            assert db is db_out
        atol, rtol = _grad_tolerance(policy, db.dtype)
        _assert_within(
            db.float(), db_ref.float(), atol=atol, rtol=rtol, label=f"dbias {label}"
        )


def _route_inputs(S: int, C: int, docs, kind: str, *, key: int = 0):
    """``(x, w, cu, dy, b, policy)`` resolving to backward route ``kind``."""

    settings = ROUTE_SETTINGS[kind]
    x, w, cu, dy = _inputs(S, C, docs, key=key, weight_dtype=settings["weight_dtype"])
    b = (
        None
        if settings["bias_dtype"] is None
        else _bias_input(C, settings["bias_dtype"], key=key)
    )
    return x, w, cu, dy, b, settings["policy"]


def _backward(x, w, cu, dy, b, policy, **kwargs):
    """``(dx, dweight, dbias_or_None)`` of one backward call."""

    grads = cake_causal_conv1d_train_backward(
        x, w, cu, dy, bias=b, silu_grad_preactivation=policy, **kwargs
    )
    return grads[0], grads[1], (grads[2] if b is not None else None)


def _fallback_of_record(cv: int, role: str) -> int:
    """The role's fallback at 128 threads in exact arithmetic: of every multiple of 32 (and, for the forward, every
    divisor of cv from 40 to 128), the highest lane use cv / (n_ch_tiles x ct_vec) x (ct_vec x (128 // ct_vec) /
    128), n_ch_tiles = ceil(cv / ct_vec), the larger on a tie."""

    candidates = set(range(32, 129, 32))
    if role == "fwd":
        candidates |= {d for d in range(40, 129) if cv % d == 0}
    return max(
        candidates,
        key=lambda d: (Fraction(cv * (128 // d), -(-cv // d) * 128), d),
    )


def test_channel_split_literal_table() -> None:
    """The host tile rule of each role against the literal table of CONTRACT §3 (the forward's, the backward's and
    the legacy forms), and its invariants over every multiple of eight up to 8192."""

    geometry = {"threads": 128, "vec": 8}
    for C, (fwd, bwd, legacy) in CHANNEL_SPLIT_TABLE.items():
        assert loader.channel_split(C, geometry=geometry, role="fwd") == fwd, C
        assert loader.channel_split(C, geometry=geometry, role="bwd") == bwd, C
        assert loader.channel_split(C, geometry=geometry, legacy=True) == legacy, C
        for role in ("fwd", "bwd"):
            assert (
                loader.channel_split(C, geometry=geometry, role=role, legacy=True)
                == legacy
            ), (C, role)
    for C in range(8, 8193, 8):
        cv = C // 8
        divisors = [d for d in range(32, 129, 32) if cv % d == 0]
        splits = {
            role: loader.channel_split(C, geometry=geometry, role=role)
            for role in ("fwd", "bwd")
        }
        splits["legacy"] = loader.channel_split(C, geometry=geometry, legacy=True)
        for key, (ct_vec, n_sub, n_ch_tiles) in splits.items():
            assert 1 <= ct_vec <= 128 and n_sub == 128 // ct_vec, (C, key)
            assert n_ch_tiles == -(-cv // ct_vec), (C, key)
        assert splits["legacy"][0] == min(cv, 128), C
        if cv <= 128:
            assert splits["fwd"][0] == splits["bwd"][0] == cv, C
        elif divisors:
            # D1: a multiple of 32 dividing cv with the highest lane use, the same for both roles.
            ct_vec = max(divisors, key=lambda d: (d * (128 // d), d))
            assert splits["fwd"][0] == splits["bwd"][0] == ct_vec, C
        else:
            for role in ("fwd", "bwd"):
                assert splits[role][0] == _fallback_of_record(cv, role), (C, role)
            assert splits["bwd"][0] % 32 == 0, C
            assert splits["fwd"][0] % 32 == 0 or (
                splits["fwd"][0] >= 40 and cv % splits["fwd"][0] == 0
            ), C
    with pytest.raises(ValueError):
        loader.channel_split(12, geometry=geometry, role="bwd")
    for role in (None, "reduce", "dx"):
        with pytest.raises(ValueError):
            loader.channel_split(1280, geometry=geometry, role=role)
        with pytest.raises(ValueError):
            loader.plan(5, 1280, 3, geometry=geometry, role=role)
    assert loader.plan(5, 1280, 3, geometry=geometry, role="bwd")["ct_vec"] == 32
    assert loader.plan(5, 1344, 3, geometry=geometry, role="fwd")["ct_vec"] == 42
    assert loader.plan(5, 1344, 3, geometry=geometry, role="bwd")["ct_vec"] == 64
    assert loader.plan(5, 1280, 3, geometry=geometry, legacy=True)["ct_vec"] == 128
    assert set(loader.plan(5, 1280, 3, geometry=geometry, role="bwd")) == set(
        loader.PLAN_SCALARS
    )
    assert "flags" in loader.LOGICAL_SCALARS and "flags" not in loader.PLAN_SCALARS


# The whole-rank trim of the backward main grid (``cake_jit.main_grid``) at 128
# threads x 8 channels: ``(arch, co-resident grid, S, C) -> grid`` on the
# routes the arch's rule trims (sm_103a: ``bwd`` at 304 CTAs; sm_100a: ``bwd``
# / ``bwd_dt`` / ``bwd_fp32g`` at 296; sm_107a: none); every other route keeps
# the co-resident grid.  The rows are the kernel module's rules-test rows.
MAIN_GRID_TABLE = (
    # sm_103a, 304 CTAs: C % 32, 1 < tiles <= {32: 27, 64: 9}, S >= 98304 and
    # a last tile of at least {32: 15, 64: 49} vectors
    ("sm_103a", 304, 250340, 1160, 300),  # ct_vec 32, 5 tiles, last tile 17
    ("sm_103a", 304, 98304, 1144, 300),  # last tile 15
    ("sm_103a", 304, 98304, 1136, 304),  # last tile 14
    ("sm_103a", 304, 98303, 1160, 304),  # below the S bound
    ("sm_103a", 304, 81920, 1160, 304),
    ("sm_103a", 304, 250340, 1032, 304),  # last tile 1
    ("sm_103a", 304, 250340, 1152, 304),  # C % 32 == 0
    ("sm_103a", 304, 250340, 1184, 304),
    ("sm_103a", 304, 250340, 1656, 301),  # 7 tiles, last 15
    ("sm_103a", 304, 250340, 2168, 297),  # 9 tiles, last 15
    ("sm_103a", 304, 250340, 2160, 304),  # 9 tiles, last 14
    ("sm_103a", 304, 250340, 1416, 303),  # ct_vec 64, 3 tiles, last 49
    ("sm_103a", 304, 250340, 1352, 304),  # ct_vec 64, last 41
    ("sm_103a", 304, 250340, 2440, 300),  # ct_vec 64, 5 tiles, last 49
    ("sm_103a", 304, 250340, 2376, 304),  # ct_vec 64, 5 tiles, last 41
    ("sm_103a", 304, 250340, 2928, 304),  # ct_vec 128: never
    ("sm_103a", 304, 250340, 1344, 304),  # C % 32 == 0
    ("sm_103a", 304, 250340, 2048, 304),  # 2 tiles divide the grid
    ("sm_103a", 304, 250340, 1000, 304),  # one tile
    ("sm_103a", 304, 250340, 6904, 297),  # ct_vec 32, 27 tiles, last 31
    ("sm_103a", 304, 250340, 4344, 289),  # ct_vec 32, 17 tiles, last 31
    ("sm_103a", 304, 250340, 7416, 304),  # ct_vec 32, 29 tiles: past the cap
    ("sm_103a", 304, 250340, 7928, 304),  # ct_vec 32, 31 tiles
    ("sm_103a", 304, 250340, 8392, 304),  # ct_vec 32, 33 tiles
    ("sm_103a", 304, 250340, 4488, 297),  # ct_vec 64, 9 tiles, last 49
    ("sm_103a", 304, 250340, 5512, 304),  # ct_vec 64, 11 tiles
    ("sm_103a", 296, 250340, 1160, 296),  # another co-resident grid
    ("sm_103a", 300, 250340, 1160, 300),
    ("sm_103a", 608, 250340, 1160, 608),
    # sm_100a, 296 CTAs: the classes (32, 5) from S 147456 with a last tile of
    # at least 27, (32, 7) from 65536 with 31, (64, 3) from 147456 with 59
    ("sm_100a", 296, 250340, 1240, 295),  # ct_vec 32, 5 tiles, last 27
    ("sm_100a", 296, 147456, 1264, 295),  # last 30, the class's S
    ("sm_100a", 296, 458752, 1272, 295),  # last 31
    ("sm_100a", 296, 147455, 1272, 296),  # below the class's S
    ("sm_100a", 296, 458752, 1232, 296),  # last 26
    ("sm_100a", 296, 250340, 1160, 296),  # last 17
    ("sm_100a", 296, 250340, 1248, 296),  # C % 32 == 0
    ("sm_100a", 296, 65536, 1784, 294),  # ct_vec 32, 7 tiles, last 31
    ("sm_100a", 296, 65535, 1784, 296),  # below the class's S
    ("sm_100a", 296, 250340, 1776, 296),  # 7 tiles, last 30
    ("sm_100a", 296, 250340, 1656, 296),  # 7 tiles, last 15
    ("sm_100a", 296, 250340, 2680, 296),  # 11 tiles: no class
    ("sm_100a", 296, 250340, 2168, 296),  # 9 tiles: no class
    ("sm_100a", 296, 147456, 1496, 294),  # ct_vec 64, 3 tiles, last 59
    ("sm_100a", 296, 458752, 1528, 294),  # last 63
    ("sm_100a", 296, 147455, 1528, 296),  # below the class's S
    ("sm_100a", 296, 458752, 1488, 296),  # last 58
    ("sm_100a", 296, 250340, 1416, 296),  # last 49
    ("sm_100a", 296, 250340, 1504, 296),  # C % 32 == 0
    ("sm_100a", 296, 250340, 2440, 296),  # ct_vec 64, 5 tiles: no class
    ("sm_100a", 296, 250340, 2928, 296),  # ct_vec 128: no class
    ("sm_100a", 304, 250340, 1272, 304),  # another co-resident grid
    # sm_107a: no class
    ("sm_107a", 424, 250340, 1416, 424),
    ("sm_107a", 424, 65536, 1528, 424),
    ("sm_107a", 424, 250340, 1160, 424),
    ("sm_107a", 424, 250340, 2440, 424),
    ("sm_107a", 296, 250340, 1160, 296),
)
MAIN_GRID_TRIMMED_ROUTES = {
    "sm_100a": ("bwd", "bwd_dt", "bwd_fp32g"),
    "sm_103a": ("bwd",),
    "sm_107a": (),
}


def test_main_grid_literal_table() -> None:
    """The backward main grid's whole-rank trim against the literal table, per
    arch and route; ``grid_mult`` above 1 or a call naming no row keeps the
    co-resident grid times ``grid_mult``."""

    geometry = {"threads": 128, "vec": 8}
    assert loader.TILE_ALIGNED_GRID == {"sm_100a": 0, "sm_103a": 304, "sm_107a": 0}
    assert loader.TILE_ALIGNED_ROUTES == ("bwd",)
    assert loader.TILE_ALIGNED_MIN_S == 98304
    assert loader.TILE_ALIGNED_MIN_LAST == {32: 15, 64: 49}
    assert loader.TILE_ALIGNED_MAX_TILES == {32: 27, 64: 9}
    assert loader.TILE_ALIGNED_CLASSES == {
        "sm_100a": {
            (296, 32, 5): (147456, 27),
            (296, 32, 7): (65536, 31),
            (296, 64, 3): (147456, 59),
        },
        "sm_103a": {},
        "sm_107a": {},
    }
    assert loader.TILE_ALIGNED_CLASS_ROUTES == {
        "sm_100a": ("bwd", "bwd_dt", "bwd_fp32g"),
        "sm_103a": (),
        "sm_107a": (),
    }
    assert set(MAIN_GRID_TRIMMED_ROUTES) == set(loader.ARCHES)
    for arch, co_resident, S, C, grid in MAIN_GRID_TABLE:
        ct_vec, _, n_ch_tiles = loader.channel_split(C, geometry=geometry, role="bwd")
        row = {"C": C, "n_ch_tiles": n_ch_tiles, "S": S, "ct_vec": ct_vec}
        for kind in ROUTE_SETTINGS:
            trims = kind in MAIN_GRID_TRIMMED_ROUTES[arch]
            got = loader.main_grid(co_resident, arch=arch, route=kind, **row)
            assert got == (grid if trims else co_resident), (arch, kind, S, C)
            assert (
                loader.main_grid(co_resident, arch=arch, grid_mult=2, route=kind, **row)
                == 2 * co_resident
            ), (arch, kind, S, C)
            assert got % n_ch_tiles == 0 or got == co_resident, (arch, kind, S, C)
        assert loader.main_grid(co_resident, arch=arch, **row) == co_resident
        assert loader.main_grid(co_resident, arch=arch, C=C, route="bwd") == co_resident
    assert loader.main_grid(304, arch="sm_103a", grid_mult=4) == 1216
    with pytest.raises(ValueError):
        loader.main_grid(304, arch="sm_103a", grid_mult=0)


# The default backward route's program per ``(arch, S, C)`` below C = 512
# (``cake_jit.selected_modules``): the ``stage_pair`` fallback of each arch,
# except on sm_107a for 256 <= C < 512 from S = 262144 tokens, where the
# ``vec=8, min_blocks=2`` program (four-token copy stages) runs.  The rows are
# the kernel module's rules-test rows on both sides of the token and channel
# boundaries; C = 512 is the first channel-selected program of every arch.
BWD_FALLBACK = {
    "sm_100a": "cake_causal_conv1d_train_9bfa65707a49df6a48ce",
    "sm_103a": "cake_causal_conv1d_train_9bfa65707a49df6a48ce",
    "sm_107a": "cake_causal_conv1d_train_b4f7ff5a3dd9ddd2f41c",
}
BWD_R200_ST4 = "cake_causal_conv1d_train_9a07bcaa0223632640ee"
BWD_AT_512 = {
    "sm_100a": "cake_causal_conv1d_train_37e428406a0cc4a8ca1b",
    "sm_103a": "cake_causal_conv1d_train_d46036aa2bc5f90bb0f6",
    "sm_107a": "cake_causal_conv1d_train_d46036aa2bc5f90bb0f6",
}
BWD_SELECTION_TOKENS = (250340, 262143, 262144, 300007)
BWD_SELECTION_CHANNELS = (128, 248, 255, 256, 384, 504, 511, 512)


def _bwd_selection_of_record(arch: str, S: int, C: int) -> str:
    if C >= 512:
        return BWD_AT_512[arch]
    if arch == "sm_107a" and S >= 262144 and C >= 256:
        return BWD_R200_ST4
    return BWD_FALLBACK[arch]


def test_backward_selection_literal_table() -> None:
    """The default backward route's program below C = 512 against the literal
    table, per arch, S and C; the token floor needs S (``ValueError`` without
    it, on sm_107a for 256 <= C < 512 alone), and no other route or arch has
    one."""

    rule = loader.select_rule(loader.MODULES[BWD_R200_ST4])
    assert rule == {
        "channels": {"sm_107a": 256},
        "kind": "channels_tokens_at_least",
        "tokens": {"sm_107a": 262144},
    }
    assert loader.MODULES[BWD_R200_ST4]["arches"] == ["sm_107a"]
    for arch in loader.ARCHES:
        for S in BWD_SELECTION_TOKENS:
            for C in BWD_SELECTION_CHANNELS:
                got = loader.selected_modules(arch, "bwd", C=C, S=S)
                assert got == (_bwd_selection_of_record(arch, S, C),), (arch, S, C)
        for C in BWD_SELECTION_CHANNELS:
            if arch == "sm_107a" and 256 <= C < 512:
                with pytest.raises(ValueError, match="needs S"):
                    loader.selected_modules(arch, "bwd", C=C)
            else:
                got = loader.selected_modules(arch, "bwd", C=C)
                assert got == (_bwd_selection_of_record(arch, 1, C),), (arch, C)
        for kind in loader.KINDS:
            if kind == "bwd" or kind not in loader.ROUTES[arch]:
                continue
            for name in loader.route_modules(arch, kind):
                rule = loader.select_rule(loader.MODULES[name])
                channels_only = rule is None or rule["kind"] == "channels_at_least"
                assert channels_only, (arch, kind, name)


def test_backward_launches_the_whole_rank_grid(monkeypatch) -> None:
    """A backward call launches :func:`cake_jit.launch_grid` of the call's
    ``(S, C)``: the main program and its reduce see that CTA count as
    ``main_ctas`` and a partial view of its rows; on the device whose
    co-resident grid the rule was measured at, the row runs on whole ranks."""

    arch = _arch()
    device = _device()
    S, C = {"sm_100a": (250340, 1240)}.get(arch, (250340, 1160))
    x, w, cu, dy = _inputs(S, C, None)
    launched = []

    def record_launches(launches, values, *, device_index):
        launched.extend((name, dict(values), grid) for name, grid in launches)

    monkeypatch.setattr(pkg, "run_modules", record_launches)
    cake_causal_conv1d_train_backward(x, w, cu, dy)
    main = loader.selected_modules(arch, "bwd", C=C, S=S)[0]
    record = loader.MODULES[main]
    assert [name for name, _, _ in launched] == [main, record["reduce"]]
    values = launched[0][1]
    ct_vec, _, n_ch_tiles = loader.channel_split(
        C, geometry=record["grid_rule"]["geometry"], role="bwd"
    )
    co_resident = record["grid_rule"]["ctas_per_sm"] * loader.sm_count(device.index)
    want = loader.main_grid(
        co_resident,
        arch=arch,
        grid_mult=loader.grid_mult(record, arch),
        C=C,
        n_ch_tiles=n_ch_tiles,
        S=S,
        ct_vec=ct_vec,
        route="bwd",
    )
    assert loader.record_route(record, arch) == "bwd"
    assert loader.launch_grid(record, device_index=device.index, C=C, S=S) == (
        want,
        1,
        1,
    )
    assert values["main_ctas"] == want
    # The main program launches the grid the call planned; the reduce plans its own (C / 4 CTAs).
    assert launched[0][2] == (want, 1, 1) and launched[1][2] is None
    assert values["partial"].shape[0] == loader.partial_rows(want, n_ch_tiles)
    untrimmed = loader.launch_grid(record, device_index=device.index)[0]
    assert untrimmed == loader.grid_per_sm(record, arch) * loader.sm_count(device.index)
    if (arch, co_resident) in (("sm_103a", 304), ("sm_100a", 296)) and (
        loader.grid_mult(record, arch) == 1
    ):
        assert want < untrimmed and want % n_ch_tiles == 0, (arch, want, untrimmed)
    else:
        assert want == untrimmed, (arch, co_resident, want)


def test_launch_grid_plans_the_row_only_where_a_rule_may_trim(monkeypatch) -> None:
    """:func:`cake_jit.launch_grid` of a row equals :func:`cake_jit.main_grid`
    of the row's plan and route on every backward record, architecture and
    co-resident grid swept; rows :func:`cake_jit.row_may_trim` rejects (no
    rule names them) launch the untrimmed grid without planning the row."""

    planned = []
    split = loader.channel_split

    def counting_split(*args, **kwargs):
        planned.append(args)
        return split(*args, **kwargs)

    monkeypatch.setattr(loader, "channel_split", counting_split)
    tokens = (4097, 65535, 65536, 98303, 98304, 147455, 147456, 250340)
    checked = trimmed = 0
    for arch in loader.ARCHES:
        monkeypatch.setattr(loader, "device_arch", lambda device_index, arch=arch: arch)
        names = sorted(
            {name for kind in loader.ROUTES[arch] for name in loader.ROUTES[arch][kind]}
        )
        records = [
            loader.MODULES[name]
            for name in names
            if loader.MODULES[name]["grid_rule"]["kind"] == "occupancy"
        ]
        assert records, arch
        for sms in (148, 152):
            monkeypatch.setattr(loader, "sm_count", lambda device_index, sms=sms: sms)
            for record in records:
                rule = record["grid_rule"]
                co_resident = int(rule["ctas_per_sm"]) * sms
                factor = loader.grid_mult(record, arch)
                for C in range(8, 4100, 8):
                    for S in tokens:
                        del planned[:]
                        got = loader.launch_grid(record, device_index=0, C=C, S=S)[0]
                        ct_vec, _, n_ch_tiles = split(
                            C, geometry=rule["geometry"], role=record["role"]
                        )
                        want = loader.main_grid(
                            co_resident,
                            arch=arch,
                            grid_mult=factor,
                            C=C,
                            n_ch_tiles=n_ch_tiles,
                            S=S,
                            ct_vec=ct_vec,
                            route=loader.record_route(record, arch),
                        )
                        key = (arch, sms, record["library"], C, S)
                        assert got == max(1, want), key
                        may = loader.row_may_trim(
                            record, arch, co_resident, factor, C, S
                        )
                        assert bool(planned) == may, key
                        checked += 1
                        trimmed += got < co_resident * factor
    assert checked and trimmed, (checked, trimmed)


def test_route_tables() -> None:
    """The host's argument -> route map and the route vocabulary."""

    assert loader.forward_route(bias=False, bf16_taps=False) == "fwd"
    assert loader.forward_route(bias=False, bf16_taps=True) == "fwd_dt"
    assert loader.forward_route(bias=True, bf16_taps=False) == "fwd_bias"
    assert loader.forward_route(bias=True, bf16_taps=True) == "fwd_bias"
    for bias, fp32, dt, kind in (
        (False, False, False, "bwd"),
        (False, False, True, "bwd_dt"),
        (False, True, False, "bwd_fp32g"),
        (False, True, True, "bwd_fp32g"),
        (True, False, False, "bwd_bias"),
        (True, False, True, "bwd_bias"),
        (True, True, False, "bwd_bias_fp32g"),
        (True, True, True, "bwd_bias_fp32g"),
    ):
        assert (
            loader.backward_route(bias=bias, fp32_grad=fp32, bf16_taps_or_dweight=dt)
            == kind
        )
    assert set(loader.KINDS) == {"fwd", "fwd_dt", "fwd_bias", *ROUTE_SETTINGS}
    assert set(loader.REQUIRED_KINDS) == {"fwd", "bwd"}
    for kind in loader.KINDS:
        assert loader.route_role(kind) == ("fwd" if kind.startswith("fwd") else "bwd")
    with pytest.raises(ValueError):
        loader.route_role("bwd_fp16")


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


@pytest.mark.parametrize("kind", tuple(ROUTE_SETTINGS))
def test_backward_programs_switch_on_channel_count(kind) -> None:
    """Exactly one program of the route runs per call, every program of the route runs for some channel count,
    and each matches the reference on both sides of its channel threshold (and of its token floor, where its rule
    has one) with a NaN-poisoned workspace."""

    _require_route(kind)
    arch = _arch()
    names = loader.route_modules(arch, kind)
    thresholds = _backward_thresholds(arch, kind)
    if not thresholds:
        pytest.skip(
            f"the {kind} route on {arch} ships one program ({names[0]}); nothing switches"
        )
    channels = sorted({C for T in thresholds for C in (T - 8, T, T + 8)} | {8192})
    rows = [
        (S, docs, C) for S, docs in ((5, (2, 0, 3)), (1000, "perf")) for C in channels
    ]
    # A token-floored program runs from its floor only: both sides of the floor at its channel threshold.
    for T, floor in _token_floors(arch, kind).values():
        rows += [(S, "perf", C) for S in (floor - 8, floor) for C in (T - 8, T, T + 8)]
    launched = set()
    for S, docs, C in rows:
        selected = loader.selected_modules(arch, kind, C=C, S=S)
        assert len(selected) == 1, f"S={S} C={C}: {selected}"
        launched.add(selected[0])
        x, w, cu, dy, b, policy = _route_inputs(S, C, docs, kind)
        workspace = cake_causal_conv1d_train_backward_workspace(S, C, _device())
        # Poison the whole buffer (sized for the largest partial buffer over every route): the selected
        # program must rewrite every partial row (and dbias row) it reduces.
        workspace.view(torch.float32).fill_(float("nan"))
        _check(
            S,
            C,
            docs,
            x=x,
            w=w,
            cu=cu,
            dy=dy,
            b=b,
            policy=policy,
            workspace=workspace,
            label=f"S={S} C={C} docs={docs} program={selected[0]}",
        )
    assert launched == set(names), f"never launched {set(names) - launched}"


@pytest.mark.parametrize("kind", ("fwd", "fwd_dt", "fwd_bias"))
def test_forward_programs_switch_on_channel_count(kind) -> None:
    """Exactly one forward program of the route runs per call, every program of the route runs for some channel
    count, and on both sides of each channel threshold ``y`` matches the reference and equals, bit for bit, the
    route's fallback program launched on its own (every forward program rounds ``y`` as the static one does)."""

    _require_route(kind)
    arch = _arch()
    names = loader.route_modules(arch, kind)
    thresholds = _backward_thresholds(arch, kind)
    if not thresholds:
        pytest.skip(
            f"the {kind} route on {arch} ships one program ({names[0]}); nothing switches"
        )
    fallback = _fallback_program(arch, kind)
    record = loader.MODULES[fallback]
    device = _device()
    channels = sorted({C for T in thresholds for C in (T - 8, T, T + 8)} | {8192})
    launched = set()
    for S, docs in ((5, (2, 0, 3)), (1000, "perf")):
        for C in channels:
            selected = loader.selected_modules(arch, kind, C=C, S=S)
            assert len(selected) == 1, f"S={S} C={C}: {selected}"
            launched.add(selected[0])
            x, w, cu, _, b, _ = _route_inputs(S, C, docs, FORWARD_ROUTE_SETTINGS[kind])
            label = f"{kind} S={S} C={C} docs={docs} program={selected[0]}"
            y = cake_causal_conv1d_train_forward(x, w, cu, bias=b)
            _assert_within(
                y,
                reference_forward(x, w, cu, b),
                atol=Y_ATOL,
                rtol=Y_RTOL,
                label=f"y {label}",
            )
            scalars = loader.plan(
                S,
                C,
                int(cu.numel()) - 1,
                geometry=record["grid_rule"]["geometry"],
                role=record["role"],
            )
            y_fallback = torch.full_like(y, float("nan"))
            values = {
                "x": x,
                "w": w,
                "cu": cu,
                "y": y_fallback,
                "work": torch.zeros(
                    loader.counter_words(record, scalars["n_ch_tiles"]),
                    dtype=torch.int32,
                    device=device,
                ),
                **{key: int(scalars[key]) for key in loader.PLAN_SCALARS},
                "flags": pkg._flags(w, b=b),
            }
            if b is not None:
                values["b"] = b
            loader.run_module(fallback, values, device_index=device.index)
            torch.cuda.synchronize(device)
            assert torch.equal(y, y_fallback), (
                f"y of {selected[0]} differs from the fallback program {fallback}: {label}"
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
        S,
        C,
        int(cu.numel()) - 1,
        geometry=record["grid_rule"]["geometry"],
        role=record["role"],
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
        "main_ctas": grid_x,
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
    assert len(got) == 2 and got[0] is dx and got[1] is dw
    assert (
        torch.equal(out, y_ref) and torch.equal(dx, dx_ref) and torch.equal(dw, dw_ref)
    )


def test_preallocated_bias_outputs_are_written_in_place() -> None:
    _require_route("bwd_bias")
    S, C, docs = 1000, 512, "perf"
    x, w, cu, dy, b, policy = _route_inputs(S, C, docs, "bwd_bias")
    dx_ref, dw_ref, db_ref = _backward(x, w, cu, dy, b, policy)
    dx = torch.full((S, C), float("nan"), dtype=torch.bfloat16, device=_device())
    dw = torch.full((C, WIDTH), float("nan"), dtype=torch.float32, device=_device())
    db = torch.full((C,), float("nan"), dtype=torch.float32, device=_device())
    got = cake_causal_conv1d_train_backward(
        x, w, cu, dy, bias=b, dx=dx, dweight=dw, dbias=db
    )
    assert len(got) == 3 and got[0] is dx and got[1] is dw and got[2] is db
    assert (
        torch.equal(dx, dx_ref) and torch.equal(dw, dw_ref) and torch.equal(db, db_ref)
    )


def _capture_and_replay(cases, kind: str, *, key: int) -> int:
    """One forward + backward graph per shape of ``cases`` under route ``kind``, each replayed ``GRAPH_REPLAYS``
    times into NaN-filled outputs and a NaN-filled workspace; every replay equals eager.  Returns the replays."""

    device = _device()
    captured = []
    for S, C, docs in cases:
        x, w, cu, dy, b, policy = _route_inputs(S, C, docs, kind, key=key)
        eager_y = cake_causal_conv1d_train_forward(x, w, cu, bias=b)
        eager = (eager_y, *_backward(x, w, cu, dy, b, policy))
        y = torch.empty_like(x)
        dx = torch.empty_like(x)
        dw = torch.empty(C, WIDTH, dtype=w.dtype, device=device)
        db = None if b is None else torch.empty(C, dtype=b.dtype, device=device)
        workspace = cake_causal_conv1d_train_backward_workspace(S, C, device)

        def step(
            x=x,
            w=w,
            cu=cu,
            dy=dy,
            b=b,
            policy=policy,
            y=y,
            dx=dx,
            dw=dw,
            db=db,
            ws=workspace,
        ):
            cake_causal_conv1d_train_forward(x, w, cu, bias=b, out=y)
            cake_causal_conv1d_train_backward(
                x,
                w,
                cu,
                dy,
                bias=b,
                silu_grad_preactivation=policy,
                workspace=ws,
                dx=dx,
                dweight=dw,
                dbias=db,
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
        # The graph holds raw pointers to x, w, b, cu and dy: keep them alive until the replays, or the
        # allocator hands their memory to the next shape's tensors and the replay reads those instead.
        captured.append((graph, (x, w, cu, dy, b), (y, dx, dw, db), eager, workspace))
    replays = 0
    for graph, _inputs_alive, outputs, eager, workspace in captured:
        # (output, eager value, name) per produced tensor; the dbias slot is None without a bias.
        compared = [
            (got, expected, name)
            for got, expected, name in zip(
                outputs, eager, ("y", "dx", "dweight", "dbias"), strict=True
            )
            if got is not None
        ]
        for _replay in range(GRAPH_REPLAYS):
            for got, _expected, _name in compared:
                got.fill_(float("nan"))
            workspace.view(torch.float32).fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize(device)
            replays += 1
            for got, expected, name in compared:
                assert torch.equal(got, expected), (
                    f"{name} of replay {replays} differs from eager ({kind})"
                )
    return replays


def test_cuda_graph_replays_match_eager() -> None:
    """One forward + backward graph per shape, replayed four times each, equals eager (the default routes)."""

    replays = _capture_and_replay(GRAPH_CASES, "bwd", key=7)
    assert replays == GRAPH_REPLAYS * len(GRAPH_CASES) == 32


@pytest.mark.parametrize("kind", NEW_BWD_KINDS)
def test_cuda_graph_replays_match_eager_per_route(kind) -> None:
    """Graph capture and replay of the round-6 routes (BF16 gradient outs and the dbias output included)."""

    _require_route(kind)
    replays = _capture_and_replay(ROUTE_CASES, kind, key=11)
    assert replays == GRAPH_REPLAYS * len(ROUTE_CASES)


def test_backward_launches_the_selected_program_then_its_reduce(monkeypatch) -> None:
    """A backward call is two launches: the program ``C`` selects, then that record's ``reduce`` program with
    the main grid's CTA count as its ``main_ctas`` scalar, a partial view of that grid's rows and ``ceil(C / 4)``
    CTAs; the forward launches the one program its route selects for ``C``."""

    S, C, docs = 1000, 512, "perf"
    device = _device()
    x, w, cu, dy = _inputs(S, C, docs)
    launched = []

    def record_launch(name, values, *, device_index, grid=None):
        launched.append((name, dict(values), device_index, grid))

    def record_launches(launches, values, *, device_index):
        for name, grid in launches:
            record_launch(name, values, device_index=device_index, grid=grid)

    # The backward issues its pair through run_modules, the forward its program through run_module.
    monkeypatch.setattr(pkg, "run_module", record_launch)
    monkeypatch.setattr(pkg, "run_modules", record_launches)
    cake_causal_conv1d_train_backward(x, w, cu, dy)
    arch = _arch()
    main = loader.selected_modules(arch, "bwd", C=C, S=S)[0]
    record = loader.MODULES[main]
    assert [name for name, _, _, _ in launched] == [main, record["reduce"]]
    assert {index for _, _, index, _ in launched} == {device.index}
    values, reduce_values = launched[0][1], launched[1][1]
    assert reduce_values == values, "both launches see one argument set"
    grid_x = loader.launch_grid(record, device_index=device.index)[0]
    # The main program launches the grid the call planned (run_module does not plan it again).
    assert launched[0][3] == loader.launch_grid(
        record, device_index=device.index, C=C, S=S
    )
    assert launched[1][3] is None
    _, _, n_ch_tiles = loader.channel_split(
        C, geometry=record["grid_rule"]["geometry"], role=record["role"]
    )
    assert values["main_ctas"] == grid_x
    assert tuple(values["partial"].shape) == (
        loader.partial_rows(grid_x, n_ch_tiles),
        C,
        WIDTH,
    )
    assert {"dw", "partial", "C", "n_ch_tiles", "ct_vec"} <= set(values)
    reduce = loader.MODULES[record["reduce"]]
    assert loader.launch_grid(reduce, device_index=device.index, C=C) == (C // 4, 1, 1)
    with pytest.raises(ValueError, match="needs the channel count"):
        loader.launch_grid(reduce, device_index=device.index)
    launched.clear()
    cake_causal_conv1d_train_forward(x, w, cu)
    assert [name for name, _, _, _ in launched] == list(
        loader.selected_modules(arch, "fwd", C=C, S=S)
    )
    # The forward passes its token-chunk counters: int32, at least two words per channel tile of its plan.
    forward = loader.MODULES[launched[0][0]]
    assert launched[0][3] == loader.launch_grid(
        forward, device_index=device.index, C=C, S=S
    )
    work = launched[0][1]["work"]
    _, _, fwd_tiles = loader.channel_split(
        C, geometry=forward["grid_rule"]["geometry"], role="fwd"
    )
    assert work.dtype == torch.int32 and work.numel() >= 2 * fwd_tiles
    assert loader.counter_words(forward, fwd_tiles) == 2 * fwd_tiles


def test_run_module_names_a_missing_argument() -> None:
    """``run_module`` refuses ``values`` that lack a logical name its module binds, naming that name and the
    kernel parameter, before it loads or launches anything (here with the caller's planned grid)."""

    name = next(
        name
        for name, record in loader.MODULES.items()
        if record["role"] == "bwd" and record["workspace_rule"] is not None
    )
    first = loader.MODULES[name]["arg_plan"][0][1]
    logical = loader.MODULES[name]["logical"][first]
    with pytest.raises(
        ValueError, match=rf"needs '{logical}' \(kernel parameter '{first}'\)"
    ):
        loader.run_module(name, {"C": 8, "S": 8}, device_index=0, grid=(1, 1, 1))


def test_backward_keeps_its_speed_beside_a_resident_side_kernel(
    record_property,
) -> None:
    """FlashInfer issue #7038's recipe: one 1-thread kernel resident on another stream must not stall the
    backward.

    The cooperative form waited for it (0.161 ms -> 17.85 ms on GB200, x28-x129 measured); the two ordinary
    launches run beside it.  The residual is one second wave: when the resident kernel displaces one main CTA,
    that CTA runs after the others and the backward takes up to twice its time alone (1.2-1.6x measured on
    GB300 with the shipped grid).  Timed with CUDA events on the current stream, median of 7 after 3 warm-ups:
    ``busy <= 2 x alone``, and ``dweight`` is bitwise equal with and without the side kernel.  Both medians
    and their ratio go to the junit properties of the test.
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
    assert busy <= 2.0 * alone, (
        f"the backward stalled behind a kernel resident on another stream: "
        f"{busy:.3f} ms beside it vs {alone:.3f} ms alone"
    )


def test_zero_tokens_return_empty_outputs() -> None:
    C = 256
    x = torch.empty(0, C, dtype=torch.bfloat16, device=_device())
    w = torch.zeros(C, WIDTH, dtype=torch.float32, device=_device())
    b = torch.zeros(C, dtype=torch.bfloat16, device=_device())
    cu = torch.zeros(2, dtype=torch.int32, device=_device())
    y = cake_causal_conv1d_train_forward(x, w, cu)
    assert tuple(y.shape) == (0, C) and y.dtype == torch.bfloat16
    dx, dw = cake_causal_conv1d_train_backward(x, w, cu, x)
    assert tuple(dx.shape) == (0, C)
    assert torch.equal(dw, torch.zeros_like(dw))
    # With a bias the empty call returns three zero gradients in the parameters' dtypes, without a launch.
    y = cake_causal_conv1d_train_forward(x, w.to(torch.bfloat16), cu, bias=b)
    assert tuple(y.shape) == (0, C)
    dx, dw, db = cake_causal_conv1d_train_backward(
        x, w, cu, x, bias=b, silu_grad_preactivation="fp32"
    )
    assert tuple(dx.shape) == (0, C) and dw.dtype == torch.float32
    assert db.dtype == torch.bfloat16 and tuple(db.shape) == (C,)
    assert torch.equal(dw, torch.zeros_like(dw)) and torch.equal(
        db, torch.zeros_like(db)
    )


@pytest.mark.parametrize("kind", tuple(ROUTE_SETTINGS))
def test_autograd_end_to_end(kind) -> None:
    """``cake_causal_conv1d`` returns the forward's bits and its backward the fused kernel's gradients, in the
    parameters' dtypes, under every route; a non-contiguous incoming gradient is materialized."""

    _require_route(kind)
    S, C, docs = 1000, 512, "perf"
    x0, w0, cu, dy, b0, policy = _route_inputs(S, C, docs, kind, key=3)
    x = x0.detach().clone().requires_grad_(True)
    w = w0.detach().clone().requires_grad_(True)
    b = None if b0 is None else b0.detach().clone().requires_grad_(True)
    y = cake_causal_conv1d(x, w, cu, b, silu_grad_preactivation=policy)
    assert torch.equal(
        y.detach(), cake_causal_conv1d_train_forward(x0, w0, cu, bias=b0)
    )
    y.backward(dy)
    dx_ref, dw_ref, db_ref = _backward(x0, w0, cu, dy, b0, policy)
    assert x.grad.dtype == torch.bfloat16 and torch.equal(x.grad, dx_ref)
    assert w.grad.dtype == w0.dtype and torch.equal(w.grad, dw_ref)
    if b is not None:
        assert b.grad.dtype == b0.dtype and torch.equal(b.grad, db_ref)
    x = x0.detach().clone().requires_grad_(True)
    w = w0.detach().clone().requires_grad_(True)
    b = None if b0 is None else b0.detach().clone().requires_grad_(True)
    cake_causal_conv1d(x, w, cu, b, silu_grad_preactivation=policy).backward(
        dy.t().contiguous().t()
    )
    assert torch.equal(x.grad, dx_ref) and torch.equal(w.grad, dw_ref)
    if b is not None:
        assert torch.equal(b.grad, db_ref)


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
        (4097, 1280, (1000, 0, 3, 3094)),
    ):
        _check(S, C, docs)
    assert loader.load.cache_info().misses == loaded_before, (
        "a changed problem size rebuilt a module"
    )


@pytest.mark.parametrize("weight_dtype", (F32, BF16), ids=_dtype_id)
@pytest.mark.parametrize("dweight_dtype", (F32, BF16), ids=_dtype_id)
@pytest.mark.parametrize("case", ROUTE_CASES, ids=_case_id)
def test_bf16_taps_and_bf16_dweight(case, weight_dtype, dweight_dtype) -> None:
    """BF16 taps (``fwd_dt`` / ``bwd_dt``) and a BF16 ``dweight`` out (``bwd_dt``), in every combination with
    FP32; the FP32 / FP32 combination is the default route and is not this test's."""

    if weight_dtype == F32 and dweight_dtype == F32:
        pytest.skip("FP32 taps and dweight run the default routes")
    _require_route("bwd_dt" if weight_dtype == BF16 else "bwd")
    _require_route("bwd_dt")
    S, C, docs = case
    x, w, cu, dy = _inputs(S, C, docs, weight_dtype=weight_dtype)
    _check(S, C, docs, x=x, w=w, cu=cu, dy=dy, dweight_dtype=dweight_dtype)


@pytest.mark.parametrize("policy", ("bf16", "fp32"))
@pytest.mark.parametrize("bias_dtype", (F32, BF16), ids=_dtype_id)
@pytest.mark.parametrize("weight_dtype", (F32, BF16), ids=_dtype_id)
@pytest.mark.parametrize("case", ROUTE_CASES, ids=_case_id)
def test_bias_routes(case, weight_dtype, bias_dtype, policy) -> None:
    """A bias in FP32 or BF16, independently of the taps' dtype, under both SiLU' policies: ``y``, ``dx``,
    ``dweight`` and ``dbias`` against the reference (the bias added after the four taps), and BF16 gradient
    outs for the BF16 parameters."""

    _require_route("bwd_bias" if policy == "bf16" else "bwd_bias_fp32g")
    S, C, docs = case
    x, w, cu, dy = _inputs(S, C, docs, weight_dtype=weight_dtype)
    b = _bias_input(C, bias_dtype)
    _check(S, C, docs, x=x, w=w, cu=cu, dy=dy, b=b, policy=policy)
    # Preallocated gradient outs in the other dtype than the parameter's.
    _check(
        S,
        C,
        docs,
        x=x,
        w=w,
        cu=cu,
        dy=dy,
        b=b,
        policy=policy,
        dweight_dtype=BF16 if weight_dtype == F32 else F32,
        dbias_dtype=BF16 if bias_dtype == F32 else F32,
    )


@pytest.mark.parametrize("weight_dtype", (F32, BF16), ids=_dtype_id)
@pytest.mark.parametrize("case", ROUTE_CASES, ids=_case_id)
def test_fp32_grad_route(case, weight_dtype) -> None:
    """``silu_grad_preactivation="fp32"`` (``bwd_fp32g``): ``dx`` within one BF16 ulp of the FP64 reference and
    ``dweight`` within the tightened bound, with FP32 and BF16 taps (the route reads the tap dtype from
    ``flags``) and a BF16 ``dweight`` out."""

    _require_route("bwd_fp32g", "bwd_dt" if weight_dtype == BF16 else "bwd")
    S, C, docs = case
    x, w, cu, dy = _inputs(S, C, docs, weight_dtype=weight_dtype)
    _check(S, C, docs, x=x, w=w, cu=cu, dy=dy, policy="fp32")
    _check(S, C, docs, x=x, w=w, cu=cu, dy=dy, policy="fp32", dweight_dtype=BF16)


def _rel_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float(
        torch.linalg.vector_norm(actual.double() - expected.double())
        / torch.linalg.vector_norm(expected.double())
    )


@pytest.mark.parametrize("case", PRECISION_CASES, ids=_case_id)
def test_fp32_grad_perf_size_precision(case, record_property) -> None:
    """The fp32 policy's perf-size precision gate: rel-L2 of ``dweight`` against the FP64 reference over the C x
    4 entries at S = 250340 is at most ``FP32G_REL_L2_GATE``; the default policy's value is reported beside it
    (information: the SiLU' at the BF16-rounded preactivation puts it in the 2.5e-5 .. 4.5e-5 class)."""

    _require_route("bwd_fp32g")
    S, C, docs = case
    x, w, cu, dy = _inputs(S, C, docs)
    _dx, dw_ref, _ = reference_backward(x, w, cu, dy, policy="fp32")
    _dx, dw, _ = _backward(x, w, cu, dy, None, "fp32")
    rel_l2 = _rel_l2(dw, dw_ref)
    _dx, dw_default, _ = _backward(x, w, cu, dy, None, "bf16")
    rel_l2_default = _rel_l2(dw_default, dw_ref)
    print(
        f"S={S} C={C}: dweight rel-L2 vs FP64 -- fp32 policy {rel_l2:.3e} (gate {FP32G_REL_L2_GATE:.1e}), "
        f"bf16 policy {rel_l2_default:.3e}"
    )
    record_property("dweight_rel_l2_fp32", rel_l2)
    record_property("dweight_rel_l2_bf16", rel_l2_default)
    assert rel_l2 <= FP32G_REL_L2_GATE, (
        f"S={S} C={C}: fp32-policy dweight rel-L2 {rel_l2:.3e} exceeds the gate {FP32G_REL_L2_GATE:.1e}"
    )
    assert rel_l2 < rel_l2_default, (
        "the fp32 policy is not more accurate than the default"
    )
    torch.cuda.empty_cache()


@pytest.mark.parametrize("kind", NEW_BWD_KINDS)
def test_nan_poisoned_workspace_per_route(kind) -> None:
    """Every round-6 backward program rewrites every partial row it reduces, the dbias slab included: a
    NaN-filled workspace (sized for the whole architecture) leaves finite gradients that match the reference
    and the gradients of a fresh workspace."""

    _require_route(kind)
    for S, C, docs in ROUTE_CASES:
        x, w, cu, dy, b, policy = _route_inputs(S, C, docs, kind)
        workspace = cake_causal_conv1d_train_backward_workspace(S, C, _device())
        workspace.view(torch.float32).fill_(float("nan"))
        _check(
            S, C, docs, x=x, w=w, cu=cu, dy=dy, b=b, policy=policy, workspace=workspace
        )
        fresh = _backward(x, w, cu, dy, b, policy)
        workspace.view(torch.float32).fill_(float("nan"))
        poisoned = _backward(x, w, cu, dy, b, policy, workspace=workspace)
        for got, expected, name in zip(
            poisoned, fresh, ("dx", "dweight", "dbias"), strict=True
        ):
            if expected is None:
                continue
            assert torch.isfinite(got.float()).all(), (kind, name)
            assert torch.equal(got, expected), (kind, S, C, name)


@pytest.mark.parametrize("kind", NEW_BWD_KINDS)
def test_gradients_are_bitwise_reproducible_per_route(kind) -> None:
    """Three launches of each round-6 route give bitwise equal ``dx``, ``dweight`` and ``dbias``."""

    _require_route(kind)
    for S, C, docs in ROUTE_CASES:
        x, w, cu, dy, b, policy = _route_inputs(S, C, docs, kind)
        first = _backward(x, w, cu, dy, b, policy)
        for _repeat in range(2):
            again = _backward(x, w, cu, dy, b, policy)
            for got, expected in zip(again, first, strict=True):
                if expected is not None:
                    assert torch.equal(got, expected), (kind, S, C)


@pytest.mark.parametrize("case", ROUTE_CASES, ids=_case_id)
def test_dt_route_with_f32_taps_matches_default_route_bitwise(case) -> None:
    """The ``bwd_dt`` route with FP32 taps (selected by a BF16 ``dweight`` out, ``flags`` bit 1 only) gives the
    default route's ``dx`` bits: the tap load form is the only difference.  The route's fallback program
    launched directly with ``flags`` 0 on its own grid gives the default fallback's ``dw`` bits too when the
    grids are equal (reported otherwise: the fixed-order partition follows each grid)."""

    _require_route("bwd_dt")
    S, C, docs = case
    arch, device = _arch(), _device()
    x, w, cu, dy = _inputs(S, C, docs)
    dx, dw = cake_causal_conv1d_train_backward(x, w, cu, dy)
    dx_dt, dw_dt = cake_causal_conv1d_train_backward(
        x, w, cu, dy, dweight=torch.empty(C, WIDTH, dtype=BF16, device=device)
    )
    assert dw_dt.dtype == BF16
    assert torch.equal(dx_dt, dx), f"bwd_dt dx differs from bwd dx at S={S} C={C}"
    atol, rtol = _grad_tolerance("bf16", BF16)
    _assert_within(
        dw_dt.float(),
        dw,
        atol=atol,
        rtol=rtol,
        label=f"bf16 dweight vs the f32 one S={S} C={C}",
    )
    # The dt fallback with flags 0 and an FP32 dw against the default fallback.
    results = {}
    for kind in ("bwd", "bwd_dt"):
        name = _fallback_program(arch, kind)
        record = loader.MODULES[name]
        scalars = loader.plan(
            S,
            C,
            int(cu.numel()) - 1,
            geometry=record["grid_rule"]["geometry"],
            role=record["role"],
        )
        grid_x = loader.launch_grid(record, device_index=device.index)[0]
        rows = loader.partial_rows(grid_x, scalars["n_ch_tiles"])
        dx_f = torch.full_like(dx, float("nan"))
        dw_f = torch.full((C, WIDTH), float("nan"), dtype=F32, device=device)
        partial = torch.full(
            (rows, C, record["workspace_rule"]["taps"]),
            float("nan"),
            dtype=F32,
            device=device,
        )
        values = {
            "x": x,
            "w": w,
            "cu": cu,
            "dy": dy,
            "dx": dx_f,
            "dw": dw_f,
            "partial": partial,
            **{key: int(scalars[key]) for key in loader.PLAN_SCALARS},
            "flags": 0,
            "main_ctas": grid_x,
        }
        loader.run_module(name, values, device_index=device.index)
        loader.run_module(record["reduce"], values, device_index=device.index)
        torch.cuda.synchronize(device)
        results[kind] = (dx_f, dw_f, grid_x)
    assert torch.equal(results["bwd_dt"][0], results["bwd"][0]), (
        "fallback dx differs across routes"
    )
    if results["bwd_dt"][2] == results["bwd"][2]:
        assert torch.equal(results["bwd_dt"][1], results["bwd"][1]), (
            f"dw of the bwd_dt fallback differs from the bwd fallback at an equal grid, S={S} C={C}"
        )
    else:
        print(
            f"S={S} C={C}: bwd_dt fallback grid {results['bwd_dt'][2]} != bwd fallback grid "
            f"{results['bwd'][2]}; dw bits are not compared"
        )


def test_workspace_bytes_and_layout() -> None:
    device = _device()
    arch = _arch()
    # The capacity owner: the backward program of the largest grid_mult x ctas_per_sm x taps over every
    # backward route sizes the public buffer (every owner shares the geometry, so their rows differ by
    # grid_mult x ctas_per_sm alone -- grid_mult over-decomposes the token streams and the partial rows grow
    # with it -- and their bytes by that times taps).
    record = loader.workspace_record(arch)
    owners = loader.workspace_owners(arch)
    assert record in owners

    def capacity_key(owner):
        return loader.grid_per_sm(owner, arch) * owner["workspace_rule"]["taps"]

    assert capacity_key(record) == max(capacity_key(owner) for owner in owners)
    geometry = record["grid_rule"]["geometry"]
    assert all(owner["grid_rule"]["geometry"] == geometry for owner in owners)
    grid = loader.launch_grid(record, device_index=device.index)[0]
    assert grid == loader.grid_mult(record, arch) * record["grid_rule"][
        "ctas_per_sm"
    ] * loader.sm_count(device.index)
    assert loader.grid_mult(record, arch) >= 1
    fwd = loader.MODULES[loader.route_modules(arch, "fwd")[0]]
    assert loader.grid_mult(fwd, arch) == 1, "co_resident grids never multiply"
    expected_owners = [
        loader.MODULES[name]
        for kind in loader.backward_kinds(arch)
        for name in loader.route_modules(arch, kind)
    ]
    assert list(owners) == expected_owners
    for C in (128, 136, 1280, 8192):
        small = cake_causal_conv1d_train_backward_workspace_bytes(1, C, device)
        large = cake_causal_conv1d_train_backward_workspace_bytes(262401, C, device)
        assert small == large > 0, "the workspace is independent of S"
        layout = loader.workspace_layout(record, C, device_index=device.index)
        _, _, n_ch_tiles = loader.channel_split(
            C, geometry=geometry, role=record["role"]
        )
        assert layout["grid"] == grid
        assert (
            layout["rows"]
            == loader.partial_rows(grid, n_ch_tiles)
            == -(-grid // n_ch_tiles)
        )
        assert layout["taps"] == record["workspace_rule"]["taps"]
        assert small == layout["total_bytes"] == 4 * layout["taps"] * C * layout["rows"]
        assert small == max(
            loader.workspace_layout(owner, C, device_index=device.index)["total_bytes"]
            for owner in owners
        )
        # The program each route selects for C writes the leading rows of its own grid inside that buffer,
        # never more than the capacity owner's bytes.
        for kind in loader.backward_kinds(arch):
            selected = loader.MODULES[loader.selected_modules(arch, kind, C=C)[0]]
            rows = loader.partial_rows(
                loader.launch_grid(selected, device_index=device.index)[0], n_ch_tiles
            )
            assert loader.partial_bytes(selected, C, rows) <= small, (C, kind)
            assert selected["workspace_rule"]["taps"] == (
                WIDTH + 1 if "bias" in kind else WIDTH
            ), kind
    # Both sides of every threshold of every route (and of every token floor) stay within the capacity.
    for kind in loader.backward_kinds(arch):
        token_counts = sorted(
            {1, *(floor for _, floor in _token_floors(arch, kind).values())}
        )
        for T in _backward_thresholds(arch, kind):
            for C, S in itertools.product((T - 8, T, 8192), token_counts):
                selected = loader.MODULES[
                    loader.selected_modules(arch, kind, C=C, S=S)[0]
                ]
                _, _, n_ch_tiles = loader.channel_split(
                    C,
                    geometry=selected["grid_rule"]["geometry"],
                    role=selected["role"],
                )
                rows = loader.partial_rows(
                    loader.launch_grid(selected, device_index=device.index)[0],
                    n_ch_tiles,
                )
                assert loader.partial_bytes(
                    selected, C, rows
                ) <= cake_causal_conv1d_train_backward_workspace_bytes(1, C, device), (
                    kind,
                    C,
                    selected["library"],
                )
    # One channel tile holds up to threads x vec channels (a partial row per CTA): 1024 for the backward's eight
    # channels per thread; 8192 channels are eight tiles; C = 136 is 17 channel vectors (seven sub-tiles of 17).
    vec, threads = geometry["vec"], geometry["threads"]
    assert loader.channel_split(threads * vec, geometry=geometry, role="bwd") == (
        threads,
        1,
        1,
    )
    assert loader.channel_split(8192, geometry=geometry, role="bwd") == (
        threads,
        1,
        8192 // (threads * vec),
    )
    assert loader.channel_split(136, geometry=geometry, role="bwd") == (
        136 // vec,
        threads // (136 // vec),
        1,
    )
    workspace = cake_causal_conv1d_train_backward_workspace(65536, 512, device)
    assert (
        workspace.dtype == torch.uint8
        and workspace.numel()
        == cake_causal_conv1d_train_backward_workspace_bytes(1, 512, device)
    )
    with pytest.raises(ValueError):
        cake_causal_conv1d_train_backward_workspace_bytes(-1, 512, device)
    with pytest.raises(ValueError):
        cake_causal_conv1d_train_backward_workspace_bytes(1, 130, device)


# Forward workspace rows: whole-row and multi-sub-tile tiles, the forward's fallback tiles (1344, 1600), a
# 2560-channel plan of five tiles and the S = 5 row below one stage.
FORWARD_WORKSPACE_CASES = (
    (1000, 512, "perf"),
    (4097, 1280, (1000, 0, 3, 3094)),
    (4097, 1344, (1000, 0, 3, 3094)),
    (1000, 1600, "perf"),
    (4097, 2560, (1000, 0, 3, 3094)),
    (65, 256, (64, 1)),
    (5, 1536, (2, 0, 3)),
)
# The backward route settings whose inputs select each forward route.
FORWARD_ROUTE_SETTINGS = {"fwd": "bwd", "fwd_dt": "bwd_dt", "fwd_bias": "bwd_bias"}


def _forward_tiles(C: int) -> int:
    """The forward's channel tiles for ``C`` at the forward records' geometry."""

    record = loader.MODULES[loader.route_modules(_arch(), "fwd")[0]]
    return loader.channel_split(
        C, geometry=record["grid_rule"]["geometry"], role="fwd"
    )[2]


def test_forward_workspace_bytes() -> None:
    """Two int32 counters per channel tile of the forward's plan, the most over every forward route; independent
    of ``S``; the allocated buffer is zero."""

    device = _device()
    arch = _arch()
    assert loader.forward_kinds(arch) == ("fwd", "fwd_dt", "fwd_bias")
    geometries = set()
    for kind in loader.forward_kinds(arch):
        for name in loader.route_modules(arch, kind):
            record = loader.MODULES[name]
            assert record["workspace_rule"] == {
                "kind": "tile_counters",
                "words_per_tile": 2,
            }, name
            geometries.add(tuple(sorted(record["grid_rule"]["geometry"].items())))
    assert len(geometries) == 1
    for C in (8, 128, 136, 1024, 1160, 1280, 1344, 1600, 2560, 3200, 8192):
        tiles = _forward_tiles(C)
        assert loader.forward_workspace_words(arch, C) == 2 * tiles, C
        small = cake_causal_conv1d_train_forward_workspace_bytes(1, C, device)
        large = cake_causal_conv1d_train_forward_workspace_bytes(262401, C, device)
        assert small == large == 4 * 2 * tiles, C
    workspace = cake_causal_conv1d_train_forward_workspace(65536, 2560, device)
    assert workspace.dtype == torch.int32 and workspace.device == device
    assert workspace.numel() == 2 * _forward_tiles(2560) and workspace.is_contiguous()
    assert int(torch.count_nonzero(workspace)) == 0
    with pytest.raises(ValueError):
        cake_causal_conv1d_train_forward_workspace_bytes(-1, 512, device)
    with pytest.raises(ValueError):
        cake_causal_conv1d_train_forward_workspace_bytes(1, 130, device)


def test_forward_reuses_one_workspace_over_calls_and_routes() -> None:
    """One caller-owned buffer serves repeated calls of every forward route and channel count in turn: every launch
    leaves the counters zero, and ``y`` is within the reference and bitwise the call without a workspace."""

    device = _device()
    workspace = cake_causal_conv1d_train_forward_workspace(
        1,
        max(
            FORWARD_WORKSPACE_CASES,
            key=lambda case: cake_causal_conv1d_train_forward_workspace_bytes(
                1, case[1], device
            ),
        )[1],
        device,
    )
    problems = []
    for kind, settings in FORWARD_ROUTE_SETTINGS.items():
        _require_route(settings)
        for S, C, docs in FORWARD_WORKSPACE_CASES:
            x, w, cu, _, b, _ = _route_inputs(S, C, docs, settings, key=13)
            assert (
                loader.forward_route(
                    bias=b is not None, bf16_taps=w.dtype == torch.bfloat16
                )
                == kind
            )
            problems.append((kind, S, C, x, w, cu, b, reference_forward(x, w, cu, b)))
    for repeat in range(3):
        for kind, S, C, x, w, cu, b, expected in problems:
            label = f"repeat {repeat} {kind} S={S} C={C}"
            y = cake_causal_conv1d_train_forward(x, w, cu, bias=b, workspace=workspace)
            torch.cuda.synchronize(device)
            assert int(torch.count_nonzero(workspace)) == 0, label
            _assert_within(y, expected, atol=Y_ATOL, rtol=Y_RTOL, label=f"y {label}")
            assert torch.equal(y, cake_causal_conv1d_train_forward(x, w, cu, bias=b)), (
                label
            )


def test_forward_on_two_streams_with_two_workspaces() -> None:
    """Two streams, each with its own buffer (or the package's buffer of each stream), run the forward at once:
    every ``y`` is bitwise the one-stream result and every buffer ends zero."""

    device = _device()
    S, C, docs = 131072, 1280, "perf"
    x, w, cu, _ = _inputs(S, C, docs, key=17)
    expected = cake_causal_conv1d_train_forward(x, w, cu)
    torch.cuda.synchronize(device)
    streams = [torch.cuda.Stream(device=device) for _ in range(2)]
    assert streams[0].cuda_stream != streams[1].cuda_stream
    workspaces = [
        cake_causal_conv1d_train_forward_workspace(S, C, device) for _ in streams
    ]
    calls = 6
    for owned in (True, False):
        outs = [[torch.empty_like(x) for _ in range(calls)] for _ in streams]
        for stream in streams:
            stream.wait_stream(torch.cuda.current_stream(device))
        for i in range(calls):
            for s, stream in enumerate(streams):
                with torch.cuda.stream(stream):
                    cake_causal_conv1d_train_forward(
                        x,
                        w,
                        cu,
                        out=outs[s][i],
                        workspace=workspaces[s] if owned else None,
                    )
        torch.cuda.synchronize(device)
        for s in range(len(streams)):
            for i in range(calls):
                assert torch.equal(outs[s][i], expected), (owned, s, i)
        held = (
            workspaces
            if owned
            else [
                pkg._FORWARD_WORKSPACES[(device.index, stream.cuda_stream)]
                for stream in streams
            ]
        )
        assert held[0].data_ptr() != held[1].data_ptr()
        assert all(int(torch.count_nonzero(ws)) == 0 for ws in held), owned


def test_forward_workspace_cache_per_device_and_stream() -> None:
    """A call without a workspace uses the zeroed buffer the package keeps for its device and current stream:
    reused by the stream's later calls, replaced by a larger zeroed one when a call needs more channel tiles (and
    kept for the smaller ones after it), and never shared by two streams."""

    device = _device()
    cache = pkg._FORWARD_WORKSPACES
    small = _inputs(1000, 512, "perf", key=19)[:3]
    large = _inputs(4097, 2560, (1000, 0, 3, 3094), key=19)[:3]
    stream, other = torch.cuda.Stream(device=device), torch.cuda.Stream(device=device)
    key = (device.index, stream.cuda_stream)
    other_key = (device.index, other.cuda_stream)
    cache.pop(key, None)
    cache.pop(other_key, None)
    for tensor in (*small, *large):
        tensor.record_stream(stream)
    stream.wait_stream(torch.cuda.current_stream(device))
    other.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        y1 = cake_causal_conv1d_train_forward(*small)
        first = cache[key]
        assert first.dtype == torch.int32 and first.numel() == 2 * _forward_tiles(512)
        y2 = cake_causal_conv1d_train_forward(*small)
        assert cache[key] is first
        y3 = cake_causal_conv1d_train_forward(*large)
        grown = cache[key]
        assert grown is not first and grown.numel() == 2 * _forward_tiles(2560)
        y4 = cake_causal_conv1d_train_forward(*small)
        assert cache[key] is grown
    with torch.cuda.stream(other):
        y5 = cake_causal_conv1d_train_forward(*small)
    torch.cuda.synchronize(device)
    assert cache[other_key].data_ptr() != cache[key].data_ptr()
    assert int(torch.count_nonzero(cache[key])) == 0
    assert int(torch.count_nonzero(cache[other_key])) == 0
    assert torch.equal(y1, y2) and torch.equal(y1, y4) and torch.equal(y1, y5)
    _assert_within(
        y1, reference_forward(*small), atol=Y_ATOL, rtol=Y_RTOL, label="y C=512"
    )
    _assert_within(
        y3, reference_forward(*large), atol=Y_ATOL, rtol=Y_RTOL, label="y C=2560"
    )


def test_forward_cuda_graph_with_and_without_workspace() -> None:
    """A captured forward with a caller-owned buffer reuses it on every replay (left zero after each); one without
    a buffer captures a zeroed buffer of its own and leaves the package's cached buffers alone.  Replays equal
    eager, also when the two graphs replay at once on two streams."""

    device = _device()
    cache = pkg._FORWARD_WORKSPACES
    for kind, settings in FORWARD_ROUTE_SETTINGS.items():
        _require_route(settings)
        S, C, docs = 4097, 1344, (1000, 0, 3, 3094)
        x, w, cu, _, b, _ = _route_inputs(S, C, docs, settings, key=23)
        eager = cake_causal_conv1d_train_forward(x, w, cu, bias=b)
        workspace = cake_causal_conv1d_train_forward_workspace(S, C, device)
        y_owned, y_own_buffer = torch.empty_like(x), torch.empty_like(x)
        side = torch.cuda.Stream(device=device)
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):
            cake_causal_conv1d_train_forward(
                x, w, cu, bias=b, out=y_owned, workspace=workspace
            )
            cake_causal_conv1d_train_forward(x, w, cu, bias=b, out=y_own_buffer)
        torch.cuda.current_stream(device).wait_stream(side)
        torch.cuda.synchronize(device)
        before = {key: held.data_ptr() for key, held in cache.items()}
        owned, own_buffer = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        with torch.cuda.graph(owned, stream=side):
            cake_causal_conv1d_train_forward(
                x, w, cu, bias=b, out=y_owned, workspace=workspace
            )
        with torch.cuda.graph(own_buffer, stream=side):
            cake_causal_conv1d_train_forward(x, w, cu, bias=b, out=y_own_buffer)
        assert {key: held.data_ptr() for key, held in cache.items()} == before, kind
        for replay in range(GRAPH_REPLAYS):
            y_owned.fill_(float("nan"))
            y_own_buffer.fill_(float("nan"))
            owned.replay()
            own_buffer.replay()
            torch.cuda.synchronize(device)
            assert torch.equal(y_owned, eager) and torch.equal(y_own_buffer, eager), (
                kind,
                replay,
            )
            assert int(torch.count_nonzero(workspace)) == 0, (kind, replay)
        streams = [torch.cuda.Stream(device=device) for _ in range(2)]
        for stream in streams:
            stream.wait_stream(torch.cuda.current_stream(device))
        for _replay in range(GRAPH_REPLAYS):
            with torch.cuda.stream(streams[0]):
                owned.replay()
            with torch.cuda.stream(streams[1]):
                own_buffer.replay()
        torch.cuda.synchronize(device)
        assert torch.equal(y_owned, eager) and torch.equal(y_own_buffer, eager), kind
        assert int(torch.count_nonzero(workspace)) == 0, kind


def test_forward_workspace_validation() -> None:
    S, C = 4097, 1280
    x, w, cu, _ = _inputs(S, C, (1000, 0, 3, 3094))
    words = cake_causal_conv1d_train_forward_workspace_bytes(S, C, _device()) // 4
    with pytest.raises(ValueError, match="workspace holds"):
        cake_causal_conv1d_train_forward(
            x,
            w,
            cu,
            workspace=torch.zeros(words - 1, dtype=torch.int32, device=_device()),
        )
    with pytest.raises(ValueError, match="torch.int32"):
        cake_causal_conv1d_train_forward(
            x,
            w,
            cu,
            workspace=torch.zeros(4 * words, dtype=torch.uint8, device=_device()),
        )
    with pytest.raises(ValueError, match="1-D"):
        cake_causal_conv1d_train_forward(
            x,
            w,
            cu,
            workspace=torch.zeros(1, words, dtype=torch.int32, device=_device()),
        )
    with pytest.raises(ValueError, match="must live on"):
        cake_causal_conv1d_train_forward(
            x, w, cu, workspace=torch.zeros(words, dtype=torch.int32)
        )
    with pytest.raises(TypeError):
        cake_causal_conv1d_train_forward(x, w, cu, workspace=[0] * words)


def test_argument_validation() -> None:
    S, C = 64, 256
    x, w, cu, dy = _inputs(S, C, None)
    b = _bias_input(C, F32)
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
    # Per buffer: the taps are FP32 or BF16 (the binding no longer checks the untyped tap / bias pointers,
    # so the host is the only guard), as are the bias and the gradient outs; the bias is one value per channel.
    with pytest.raises(
        ValueError, match="weight must be torch.float32 or torch.bfloat16"
    ):
        cake_causal_conv1d_train_forward(x, w.half(), cu)
    with pytest.raises(
        ValueError, match="weight must be torch.float32 or torch.bfloat16"
    ):
        cake_causal_conv1d_train_backward(x, w.half(), cu, dy)
    with pytest.raises(
        ValueError, match="dweight must be torch.float32 or torch.bfloat16"
    ):
        cake_causal_conv1d_train_backward(
            x,
            w,
            cu,
            dy,
            dweight=torch.empty(C, WIDTH, dtype=torch.float16, device=_device()),
        )
    with pytest.raises(
        ValueError, match="bias must be torch.float32 or torch.bfloat16"
    ):
        cake_causal_conv1d_train_forward(x, w, cu, bias=b.half())
    with pytest.raises(
        ValueError, match="bias must be torch.float32 or torch.bfloat16"
    ):
        cake_causal_conv1d_train_backward(x, w, cu, dy, bias=b.half())
    with pytest.raises(ValueError, match=r"bias must be a \[C\]"):
        cake_causal_conv1d_train_forward(x, w, cu, bias=b[: C // 2].contiguous())
    with pytest.raises(ValueError, match=r"bias must be a \[C\]"):
        cake_causal_conv1d_train_forward(x, w, cu, bias=w[:, :1].contiguous())
    with pytest.raises(ValueError, match="contiguous"):
        cake_causal_conv1d_train_forward(x, w, cu, bias=w[:, 0])
    with pytest.raises(
        ValueError, match="dbias must be torch.float32 or torch.bfloat16"
    ):
        cake_causal_conv1d_train_backward(
            x,
            w,
            cu,
            dy,
            bias=b,
            dbias=torch.empty(C, dtype=torch.float16, device=_device()),
        )
    with pytest.raises(ValueError, match="dbias must have shape"):
        cake_causal_conv1d_train_backward(
            x, w, cu, dy, bias=b, dbias=torch.empty(C + 8, dtype=F32, device=_device())
        )
    with pytest.raises(ValueError, match="dbias"):
        cake_causal_conv1d_train_backward(x, w, cu, dy, dbias=torch.empty_like(b))
    with pytest.raises(ValueError, match="silu_grad_preactivation"):
        cake_causal_conv1d_train_backward(x, w, cu, dy, silu_grad_preactivation="fp16")
    with pytest.raises(ValueError, match="silu_grad_preactivation"):
        cake_causal_conv1d(x, w, cu, silu_grad_preactivation="bf32")
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


def test_every_route_ships_on_every_architecture() -> None:
    """The delivered package has no trap: every route of ``KINDS`` is exported on every architecture, with every
    library in ``build.json``.  Fails until the exporter has rendered the round-6 programs."""

    libraries = cake_build.libraries_by_name()
    for arch in loader.ARCHES:
        assert set(loader.ROUTES[arch]) == set(loader.KINDS), (
            arch,
            sorted(loader.ROUTES[arch]),
        )
        for kind in loader.KINDS:
            for name in loader.route_modules(arch, kind):
                library = loader.library_name(name, arch)
                assert library in libraries, (
                    f"{arch} {kind}: {library} is not in build.json (awaiting the exporter's render)"
                )


def test_registry_records_reference_the_build_manifest() -> None:
    """Every ``MODULES`` record names its ``build.json`` library and nothing to compile.

    The manifest (schema ``cake-build/1``) carries exactly one entry per (library,
    architecture) the registry references, each with the kernel and binding
    sources under ``csrc/`` of this package, their digests and a closure digest;
    the registry itself records the host plan only.
    """

    manifest = cake_build.manifest()
    assert manifest["schema"] == "cake-build/1"
    # The manifest names the package by its FlashInfer import path, also in a copy
    # vendored under another name (``pkg.__name__`` is then the copy's).
    assert manifest["package"] == "flashinfer.cake_causal_conv1d_train_backend"
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
    for arch, by_kind in loader.ROUTES.items():
        for kind, names in by_kind.items():
            assert kind in loader.KINDS, (arch, kind)
            for name in names:
                assert loader.MODULES[name]["role"] == loader.route_role(kind), (
                    arch,
                    kind,
                    name,
                )
                assert arch in loader.MODULES[name]["arches"], (arch, kind, name)
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
            assert not {"select_rule", "reduce", "grid_mult"} & (
                set(record) | set(grid)
            )
        else:
            # The forward keeps the co-resident grid it always launched normally; the backward programs
            # of the split are sized by occupancy, over-decomposed per architecture by grid_mult.
            assert (
                grid["kind"]
                == {"fwd": "co_resident", "bwd": "occupancy"}[record["role"]]
            ), name
            assert grid["ctas_per_sm"] >= 1
            if grid["kind"] == "occupancy":
                assert set(grid["grid_mult"]) == set(record["arches"]), name
                assert all(
                    isinstance(k, int) and k >= 1 for k in grid["grid_mult"].values()
                ), name
            else:
                assert "grid_mult" not in grid, name
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
        # A bias backward program maps b and carries the five-slab workspace rule whose dbias slab its reduce
        # program sums into db; a dtype_flags program maps flags (its reduce stores dw / db in the dtypes
        # flags names); the default routes' programs map neither.  No main program maps dw or db: the
        # reduce program writes them from the partial rows.
        logical = set(record["logical"].values())
        reduce_core = {"partial", "dw", "C", "main_ctas", "n_ch_tiles", "ct_vec"}
        if record["role"] == "bwd":
            assert record["workspace_rule"]["kind"] == "partial_rows", name
            assert not {"dw", "db"} & logical, name
            reduce_logical = set(loader.MODULES[record["reduce"]]["logical"].values())
            assert record["workspace_rule"]["taps"] == (
                WIDTH + 1 if "b" in logical else WIDTH
            ), name
            assert ("b" in logical) == ("db" in reduce_logical), name
            assert ("flags" in logical) == ("flags" in reduce_logical), name
            assert reduce_core <= reduce_logical <= reduce_core | {"db", "flags"}, name
        elif record["role"] == "reduce":
            assert record["workspace_rule"] is None, name
            assert reduce_core <= logical <= reduce_core | {"db", "flags"}, name
        else:
            # A forward program takes its token chunks from the ``work`` counters: two int32 words per channel
            # tile (the next-chunk counter and the done-CTA counter).
            assert record["workspace_rule"] == {
                "kind": "tile_counters",
                "words_per_tile": 2,
            }, name
            assert "work" in logical and not {"dw", "db", "partial"} & logical, name
        assert (
            record["workspace_rule"] is None
            or record["workspace_rule"]["kind"] in loader.WORKSPACE_KINDS
        ), name
        if "b" in logical:
            assert "flags" in logical, name
        # The selection vocabulary: a forward or backward rule is ``channels_at_least`` with one positive
        # multiple-of-8 threshold per architecture it names, or ``channels_tokens_at_least`` (a backward rule)
        # with a positive token floor beside each threshold as well.
        rule = loader.select_rule(record)
        if rule is not None:
            assert record["role"] in ("fwd", "bwd"), name
            assert rule["kind"] in loader.SELECT_KINDS, name
            if rule["kind"] == "channels_tokens_at_least":
                assert record["role"] == "bwd", name
                assert set(rule) == {"kind", "channels", "tokens"}, name
                assert set(rule["tokens"]) == set(rule["channels"]), name
                assert all(tokens > 0 for tokens in rule["tokens"].values()), name
            else:
                assert set(rule) == {"kind", "channels"}, name
            assert set(rule["channels"]) <= set(record["arches"]), name
            for threshold in rule["channels"].values():
                assert threshold > 0 and threshold % loader.CHANNEL_MULTIPLE == 0, name
        for arch in record["arches"]:
            library = loader.library_name(name, arch)
            referenced.add(library)
            assert library in libraries, (
                f"{library} is not in build.json (awaiting the exporter's render)"
            )
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
    # Per architecture and route, forward or backward: exactly one fallback record, every routed record's
    # rule names the architecture with its own threshold (distinct thresholds, at most three routed
    # programs without a token floor and at most one with one), one program selected per channel count and
    # token count -- the highest threshold at most C whose token floor (if any) S meets, the fallback below
    # the lowest.
    for arch in loader.ARCHES:
        for kind in loader.KINDS:
            if kind not in loader.ROUTES[arch]:
                continue
            names = loader.route_modules(arch, kind)
            rules = {name: loader.select_rule(loader.MODULES[name]) for name in names}
            fallbacks = [name for name, rule in rules.items() if rule is None]
            assert len(fallbacks) == 1, (arch, kind, fallbacks)
            positives = sorted(
                (int(rule["channels"][arch]), name)
                for name, rule in rules.items()
                if rule is not None
            )
            assert all(
                arch in rules[name]["channels"]
                for name in names
                if rules[name] is not None
            ), (arch, kind, names)
            thresholds = [threshold for threshold, _ in positives]
            assert thresholds == sorted(set(thresholds)), (arch, kind, positives)
            floors = _token_floors(arch, kind)
            assert len(positives) - len(floors) <= 3, (arch, kind, positives)
            assert len(floors) <= 1, (arch, kind, floors)
            for S in sorted({1, *(floor for _, floor in floors.values())}):
                previous = fallbacks[0]
                for threshold, name in positives:
                    at = loader.selected_modules(arch, kind, C=threshold, S=S)
                    if name in floors and floors[name][1] > S:
                        # Below its token floor the program never runs.
                        assert at == (previous,), (arch, kind, S, threshold)
                        continue
                    below = loader.selected_modules(arch, kind, C=threshold - 8, S=S)
                    assert at == (name,), (arch, kind, S, threshold)
                    assert below == (previous,), (arch, kind, S, threshold)
                    previous = name
                top = loader.selected_modules(arch, kind, C=8192, S=S)
                assert top == (previous,), (arch, kind, S)
            for name, (threshold, floor) in floors.items():
                # A token floor decides at its threshold: S below it runs another
                # program, and no S is refused.
                at = loader.selected_modules(arch, kind, C=threshold, S=floor)
                below = loader.selected_modules(arch, kind, C=threshold, S=floor - 1)
                assert at == (name,) and below != (name,), (arch, kind, name)
                with pytest.raises(ValueError, match="needs S"):
                    loader.selected_modules(arch, kind, C=threshold)
            for C in (8, 128, 512, 1024, 1280, 8192):
                selected = loader.selected_modules(arch, kind, C=C, S=1)
                assert len(selected) == 1 and selected[0] in names, (
                    arch,
                    kind,
                    C,
                    selected,
                )
        owners = loader.workspace_owners(arch)
        assert [owner["library"] for owner in owners] == [
            name
            for kind in loader.backward_kinds(arch)
            for name in loader.route_modules(arch, kind)
        ]
        for owner in owners:
            assert owner["role"] == "bwd"
            assert owner["workspace_rule"]["kind"] == "partial_rows"
            assert owner["workspace_rule"]["taps"] in (WIDTH, WIDTH + 1)
            assert owner["launch"]["cooperative"] is False
            assert arch in loader.MODULES[owner["reduce"]]["arches"], (
                arch,
                owner["library"],
            )
        assert loader.workspace_record(arch) in owners
    backward = loader.MODULES[_fallback_program("sm_100a")]
    assert backward["workspace_rule"] == {"kind": "partial_rows", "taps": WIDTH}
    assert backward["launch"]["cooperative"] is False
