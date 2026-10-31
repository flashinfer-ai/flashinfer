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

Tests of the non-deterministic Cake packed width-4 depthwise causal convolution + SiLU training package
(``deterministic=False``; SM100 / SM103 / SM107).

Every kernel output is compared with an independent plain-torch reference of the same math with document-local
shifts, under the deterministic package's budgets: BF16 ``y`` and ``dx`` within ``atol 1e-3 + rtol 2^-7`` (one
BF16 ulp) with no element outside; the gradients within ``atol 5e-2 + rtol 1e-4`` of the FP64 sum under the
default SiLU' policy and within ``atol 1e-3 + rtol 2e-6`` of the FP64 reference under
``silu_grad_preactivation="fp32"`` (a BF16 gradient out widened by its own rounding).  Against the deterministic
package (when importable) ``y`` and ``dx`` are bitwise equal at every case and route, and ``dweight`` / ``dbias``
are within the same budgets of the deterministic ones.  The relaxed property is the run-to-run bits of the
gradients only: >= 20 repeated launches keep ``dx`` bitwise and every gradient within budget, and the worst
case over the repeats is recorded.  The fp32 policy keeps the perf-size precision gate (rel-L2 <= 2e-7 at S =
250340, C 512 and 1280) as its worst case over the repeats.

The workspace contract: a caller-owned zeroed workspace and the package's per-(device, stream) workspace read
back all zero after every launch, over changed shapes and routes; NaN-poisoned outputs are fully overwritten;
CUDA Graph capture / replay with and without a caller-owned workspace; two streams with two workspaces at once;
the backward is ONE ordinary launch (no reduce program) and keeps its speed beside a kernel resident on another
stream (FlashInfer issues #7038 / #7548).  Autograd, non-reentrant activation checkpointing (2 x forward +
backward) and gradient accumulation; the ``deterministic`` keyword and torch's deterministic mode.

This suite names its package once, through ``tests.test_helpers.cake_package`` (key
``cake.cake_causal_conv1d_train``), and the deterministic package for the parity checks through the same table;
a vendored build without the deterministic package skips those checks.  Every test skips until the exporter has
rendered the package (``build.json`` and ``csrc/``).
"""

from __future__ import annotations

import gc
import statistics

import pytest
import torch

from tests.test_helpers.cake_package import cake_package
from tests.test_helpers.cake_causal_conv1d_early_render import apply_to_package

# Test-only: the early render's backward rules predate the required ``slots`` (the package applies no default).
apply_to_package()

KEY = "cake.cake_causal_conv1d_train"
pkg = cake_package(KEY)
cake_build = cake_package(KEY, "cake_build")
loader = cake_package(KEY, "cake_jit")

SUPPORTED_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
WIDTH = 4
F32, BF16 = torch.float32, torch.bfloat16
# The deterministic package's tolerances (the budgets this package keeps).
Y_ATOL, Y_RTOL = 1e-3, 2**-7
DW_ATOL, DW_RTOL = 5e-2, 1e-4
DW_FP32G_ATOL, DW_FP32G_RTOL = 1e-3, 2e-6
FP32G_REL_L2_GATE = 2e-7
BLOCK = 8192
COMPARE_CHUNK = 2**26
PERF_FRACTIONS = (0.37, 0.29, 0.21)
SEED = 8065
# Repeated launches per row of the run-to-run checks (the worst case over them is the reported figure).
REPEATS = 20

# (S, C, documents): one-token and short documents, empty documents before / between / after others, boundaries
# inside a sub-tile and at 64 / 65, the perf packing, 38 documents, the channel thresholds of the routed programs,
# the multi-sub-tile tiles (1280, 1536, 2560) and the partial last channel tiles (1160, 1344, 1600).
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
    (4097, 1160, (1000, 0, 3, 3094)),
    (4097, 1344, (1000, 0, 3, 3094)),
    (1000, 1600, "perf"),
)
ROUTE_CASES = (
    (1000, 512, "perf"),
    (4097, 1280, (1000, 0, 3, 3094)),
    (5, 1536, (2, 0, 3)),
    (65, 256, (64, 1)),
    (4097, 1344, (1000, 0, 3, 3094)),
)
# A large row per architecture class for the run-to-run checks (many CTAs per channel tile race on the accumulators).
REPEAT_CASES = (
    (65536, 512, "perf"),
    (65536, 1280, "perf"),
    (4097, 1160, (1000, 0, 3, 3094)),
)
PRECISION_CASES = ((250340, 512, "perf"), (250340, 1280, "perf"))
PRECISION_REPEATS = 5
# S x C > 2^31 elements (64-bit offsets) at a token count past the sm_107a token floor.
LARGE_CASE = (262401, 8192, "perf")
LARGE_CASE_FREE_BYTES = 40 * 2**30
GRAPH_CASES = (
    (65, 256, (64, 1)),
    (5, 136, (2, 0, 3)),
    (1000, 512, "perf"),
    (4097, 1280, (1000, 0, 3, 3094)),
    (4097, 2560, (1000, 0, 3, 3094)),
)
GRAPH_REPLAYS = 4
ROUTE_SETTINGS = {
    "bwd": dict(weight_dtype=F32, bias_dtype=None, policy="bf16"),
    "bwd_dt": dict(weight_dtype=BF16, bias_dtype=None, policy="bf16"),
    "bwd_bias": dict(weight_dtype=F32, bias_dtype=F32, policy="bf16"),
    "bwd_fp32g": dict(weight_dtype=F32, bias_dtype=None, policy="fp32"),
    "bwd_bias_fp32g": dict(weight_dtype=F32, bias_dtype=F32, policy="fp32"),
}
FORWARD_OF = {
    "bwd": "fwd",
    "bwd_dt": "fwd_dt",
    "bwd_bias": "fwd_bias",
    "bwd_fp32g": "fwd",
    "bwd_bias_fp32g": "fwd_bias",
}
# Every dtype combination a backward call takes: taps x bias (none, FP32, BF16) x SiLU' policy x the dweight /
# dbias out dtypes (each FP32 or BF16, independently of its parameter).  Each resolves to one route: BF16 taps on
# the fp32g and bias routes, and both mixed bwd_dt combinations (BF16 taps + FP32 dweight, FP32 taps + BF16
# dweight), are among them.
DTYPE_COMBOS = tuple(
    (weight_dtype, bias_dtype, policy, dweight_dtype, dbias_dtype)
    for policy in ("bf16", "fp32")
    for weight_dtype in (F32, BF16)
    for bias_dtype in (None, F32, BF16)
    for dweight_dtype in (F32, BF16)
    for dbias_dtype in ((None,) if bias_dtype is None else (F32, BF16))
)
# The autograd entry returns the gradients in the parameters' dtypes: taps x bias x policy.
AUTOGRAD_COMBOS = tuple(
    (weight_dtype, bias_dtype, policy)
    for policy in ("bf16", "fp32")
    for weight_dtype in (F32, BF16)
    for bias_dtype in (None, F32, BF16)
)


def _device() -> torch.device:
    return torch.device("cuda:0")


def _capability() -> tuple[int, int]:
    return torch.cuda.get_device_capability(_device())


def _supported() -> bool:
    return (
        torch.cuda.is_available()
        and _capability() in SUPPORTED_CAPABILITIES
        and pkg.is_cake_causal_conv1d_train_supported(_device())
    )


pytestmark = pytest.mark.skipif(
    not _supported(),
    reason=(
        "the non-deterministic cake_causal_conv1d_train package needs an SM100 / SM103 / SM107 device and its "
        "rendered build.json / csrc (TODO-EXPORT until the exporter renders it)"
    ),
)


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    if _supported():
        arch = loader.arch_for_capability(_capability())
        for spec in loader.specs_for_arch(arch):
            spec.build_and_load()
    yield


def _det():
    """The deterministic package (parity checks), or a skip when this build does not carry it."""

    try:
        return cake_package("cake_causal_conv1d_det_train")
    except Exception as exc:  # noqa: BLE001 - a vendored copy without the det package skips the parity checks
        pytest.skip(f"the deterministic package is not importable here: {exc}")


def _case_id(case) -> str:
    S, C, docs = case
    tag = "one" if docs is None else ("perf" if docs == "perf" else f"{len(docs)}docs")
    return f"S{S}-C{C}-{tag}"


def _dtype_id(dtype) -> str:
    return {None: "none", F32: "f32", BF16: "bf16"}[dtype]


def _combo_id(combo) -> str:
    weight_dtype, bias_dtype, policy, *outs = combo
    label = f"w{_dtype_id(weight_dtype)}-b{_dtype_id(bias_dtype)}-{policy}"
    if outs:
        dweight_dtype, dbias_dtype = outs
        label += f"-dw{_dtype_id(dweight_dtype)}"
        if bias_dtype is not None:
            label += f"-db{_dtype_id(dbias_dtype)}"
    return label


def _combo_route(weight_dtype, bias_dtype, policy, dweight_dtype=None) -> str:
    """The backward route a call of these dtypes runs (``dweight_dtype`` ``None``: the taps' dtype)."""

    return loader.backward_route(
        bias=bias_dtype is not None,
        fp32_grad=policy == "fp32",
        bf16_taps_or_dweight=BF16 in (weight_dtype, dweight_dtype or weight_dtype),
    )


def _det_or_none():
    """The deterministic package, or ``None`` when this build does not carry it (the parity checks are skipped,
    the reference checks still run)."""

    try:
        return cake_package("cake_causal_conv1d_det_train")
    except Exception:  # noqa: BLE001 - a vendored copy without the det package
        return None


def _arch() -> str:
    return loader.arch_for_capability(_capability())


def _route_pending(arch: str, kind: str) -> str | None:
    if kind not in loader.ROUTES.get(arch, {}):
        return f"route {kind!r} is not exported for {arch}"
    libraries = cake_build.libraries_by_name()
    for name in loader.route_modules(arch, kind):
        library = loader.library_name(name, arch)
        if library not in libraries:
            return f"route {kind!r}: {library} is not in build.json yet (awaiting the exporter's render)"
    return None


def _require_route(*kinds: str) -> None:
    for kind in kinds:
        for needed in (kind, FORWARD_OF.get(kind, kind)):
            pending = _route_pending(_arch(), needed)
            if pending:
                pytest.skip(pending)


def _backward_thresholds(arch: str, kind: str) -> tuple[int, ...]:
    """The channel thresholds of the programs route ``kind`` selects by channel count on ``arch`` (empty when the
    route ships its fallback program only there)."""

    thresholds = set()
    for name in loader.route_modules(arch, kind):
        rule = loader.select_rule(loader.MODULES[name])
        if rule is not None and arch in rule["channels"]:
            thresholds.add(int(rule["channels"][arch]))
    return tuple(sorted(thresholds))


def _token_floors(arch: str, kind: str) -> dict[str, tuple[int, int]]:
    """The programs of route ``kind`` on ``arch`` whose rule carries a token floor: name -> (channel threshold,
    token floor)."""

    floors = {}
    for name in loader.route_modules(arch, kind):
        rule = loader.select_rule(loader.MODULES[name])
        if rule is not None and arch in rule.get("tokens", {}):
            floors[name] = (int(rule["channels"][arch]), int(rule["tokens"][arch]))
    return floors


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
    return torch.bucketize(torch.arange(S, device=cu.device), cu[1:].long(), right=True)


def _inputs(S: int, C: int, docs, *, key: int = 0, weight_dtype=F32):
    """``(x, w, cu, dy)``: BF16 ``x`` / ``dy`` of N(0, 1), taps of N(0, 0.25) in ``weight_dtype``."""

    generator = _generator(S, C, key)
    x = torch.randn(
        S, C, dtype=torch.float32, device=_device(), generator=generator
    ).to(BF16)
    w = 0.5 * torch.randn(
        C, WIDTH, dtype=torch.float32, device=_device(), generator=generator
    )
    dy = torch.randn(
        S, C, dtype=torch.float32, device=_device(), generator=generator
    ).to(BF16)
    return x, w.to(weight_dtype), _cu_seqlens(S, docs), dy


def _bias_input(C: int, dtype, *, key: int = 0) -> torch.Tensor:
    generator = _generator(C, 97, key)
    return (
        0.5 * torch.randn(C, dtype=torch.float32, device=_device(), generator=generator)
    ).to(dtype)


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


def _shift_down(v: torch.Tensor, ids: torch.Tensor, d: int) -> torch.Tensor:
    out = torch.zeros_like(v)
    if d == 0:
        return v.clone()
    if d < v.shape[0]:
        same = (ids[d:] == ids[:-d]).unsqueeze(1)
        out[d:] = torch.where(same, v[:-d], 0.0)
    return out


def _shift_up(v: torch.Tensor, ids: torch.Tensor, d: int) -> torch.Tensor:
    out = torch.zeros_like(v)
    if d == 0:
        return v.clone()
    if d < v.shape[0]:
        same = (ids[:-d] == ids[d:]).unsqueeze(1)
        out[:-d] = torch.where(same, v[d:], 0.0)
    return out


def _preactivation(x, ids, w, b):
    pre = torch.zeros_like(x)
    for k in range(WIDTH):
        pre = pre + w[:, k].to(x.dtype) * _shift_down(x, ids, WIDTH - 1 - k)
    if b is not None:
        pre = pre + b.to(x.dtype)
    return pre


def reference_forward(x, w, cu, b=None) -> torch.Tensor:
    S = x.shape[0]
    ids = _doc_ids(S, cu)
    y = torch.empty_like(x)
    for a in range(0, S, BLOCK):
        bb = min(a + BLOCK, S)
        lo = max(a - (WIDTH - 1), 0)
        pre = _preactivation(x[lo:bb].float(), ids[lo:bb], w, b)[a - lo :]
        y[a:bb] = (pre * torch.sigmoid(pre)).bfloat16()
    return y


def reference_backward(x, w, cu, dy, *, b=None, policy="bf16"):
    """``dx`` (BF16), ``dweight`` and ``dbias`` (FP64 sums; ``dbias`` ``None`` without a bias); ``policy`` as the
    deterministic suite's reference (``"bf16"``: SiLU' at the BF16-rounded FP32 preactivation; ``"fp32"``: FP64
    throughout)."""

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
    atol, rtol = (
        (DW_ATOL, DW_RTOL) if policy == "bf16" else (DW_FP32G_ATOL, DW_FP32G_RTOL)
    )
    if dtype == torch.bfloat16:
        return atol * (1 + 2**-8), 2**-7
    return atol, rtol


def _assert_within(actual, expected, *, atol, rtol, label) -> float:
    """No element outside ``atol + rtol |expected|``; returns the worst ``|actual - expected| / bound``."""

    assert tuple(actual.shape) == tuple(expected.shape), label
    flat_actual, flat_expected = actual.reshape(-1), expected.reshape(-1)
    count, first, worst = 0, None, 0.0
    for start in range(0, flat_actual.numel(), COMPARE_CHUNK):
        a = flat_actual[start : start + COMPARE_CHUNK].double()
        b = flat_expected[start : start + COMPARE_CHUNK].double()
        assert torch.isfinite(a).all(), f"{label} has non-finite values"
        bound = atol + rtol * b.abs()
        ratio = (a - b).abs() / bound
        worst = max(worst, float(ratio.max().item()) if ratio.numel() else 0.0)
        outside = ratio > 1.0
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
    return worst


def _forward(x, w, cu, **kwargs) -> torch.Tensor:
    return pkg.cake_causal_conv1d_train_forward(x, w, cu, deterministic=False, **kwargs)


def _backward(x, w, cu, dy, b=None, policy="bf16", **kwargs):
    """``(dx, dweight, dbias_or_None)`` of one non-deterministic backward call."""

    grads = pkg.cake_causal_conv1d_train_backward(
        x,
        w,
        cu,
        dy,
        bias=b,
        silu_grad_preactivation=policy,
        deterministic=False,
        **kwargs,
    )
    return grads[0], grads[1], (grads[2] if b is not None else None)


def _det_backward(det, x, w, cu, dy, b=None, policy="bf16", **kwargs):
    grads = det.cake_causal_conv1d_det_train_backward(
        x, w, cu, dy, bias=b, silu_grad_preactivation=policy, **kwargs
    )
    return grads[0], grads[1], (grads[2] if b is not None else None)


def _check_grads(dx, dw, db, refs, *, policy, label) -> tuple[float, float]:
    """``dx`` within one BF16 ulp of the reference, ``dw`` / ``db`` within the policy's budget; returns the worst
    gradient ratios ``(dw, db)``."""

    dx_ref, dw_ref, db_ref = refs
    _assert_within(dx, dx_ref, atol=Y_ATOL, rtol=Y_RTOL, label=f"dx {label}")
    atol, rtol = _grad_tolerance(policy, dw.dtype)
    worst_dw = _assert_within(
        dw.float(), dw_ref.float(), atol=atol, rtol=rtol, label=f"dweight {label}"
    )
    worst_db = 0.0
    if db is not None:
        atol, rtol = _grad_tolerance(policy, db.dtype)
        worst_db = _assert_within(
            db.float(), db_ref.float(), atol=atol, rtol=rtol, label=f"dbias {label}"
        )
    return worst_dw, worst_db


def _stream_workspace(device: torch.device) -> torch.Tensor | None:
    """The package's backward workspace of the device's current stream (``None`` before its first call)."""

    key = (device.index, int(torch.cuda.current_stream(device).cuda_stream))
    return pkg._BACKWARD_WORKSPACES.get(key)


# ----------------------------------------------------------------------------------------------------------------
# Accuracy against the independent reference and parity with the deterministic package.


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_forward_backward_match_reference(case) -> None:
    S, C, docs = case
    x, w, cu, dy = _inputs(S, C, docs)
    y = _forward(x, w, cu)
    assert y.dtype == BF16 and tuple(y.shape) == tuple(x.shape) and y.is_contiguous()
    _assert_within(
        y, reference_forward(x, w, cu), atol=Y_ATOL, rtol=Y_RTOL, label=f"y {case}"
    )
    dx, dw, _ = _backward(x, w, cu, dy)
    assert dx.dtype == BF16 and dw.dtype == F32 and tuple(dw.shape) == (C, WIDTH)
    _check_grads(
        dx, dw, None, reference_backward(x, w, cu, dy), policy="bf16", label=str(case)
    )


@pytest.mark.parametrize("combo", DTYPE_COMBOS, ids=_combo_id)
@pytest.mark.parametrize("case", ROUTE_CASES, ids=_case_id)
def test_routes_match_reference(case, combo) -> None:
    """Every backward route in every dtype combination (FP32 / BF16 taps, no / FP32 / BF16 bias, both SiLU'
    policies, FP32 / BF16 ``dweight`` and ``dbias`` outs): ``y``, ``dx``, ``dweight`` and ``dbias`` against the
    reference; ``y`` / ``dx`` bitwise the deterministic package's for the same call and its ``dweight`` /
    ``dbias`` within the route's budget of the deterministic ones (when that package is importable)."""

    weight_dtype, bias_dtype, policy, dweight_dtype, dbias_dtype = combo
    kind = _combo_route(weight_dtype, bias_dtype, policy, dweight_dtype)
    _require_route(kind)
    S, C, docs = case
    device = _device()
    label = f"{kind} {_combo_id(combo)} {case}"
    x, w, cu, dy = _inputs(S, C, docs, key=1, weight_dtype=weight_dtype)
    b = None if bias_dtype is None else _bias_input(C, bias_dtype, key=1)
    y = _forward(x, w, cu, bias=b)
    _assert_within(
        y, reference_forward(x, w, cu, b), atol=Y_ATOL, rtol=Y_RTOL, label=f"y {label}"
    )

    def outs():
        return {
            "dweight": torch.empty(C, WIDTH, dtype=dweight_dtype, device=device),
            "dbias": None
            if b is None
            else torch.empty(C, dtype=dbias_dtype, device=device),
        }

    out = outs()
    dx, dw, db = _backward(x, w, cu, dy, b, policy, **out)
    assert dw is out["dweight"] and db is out["dbias"], label
    refs = reference_backward(x, w, cu, dy, b=b, policy=policy)
    _check_grads(dx, dw, db, refs, policy=policy, label=label)
    det = _det_or_none()
    if det is None:
        return
    assert torch.equal(y, det.cake_causal_conv1d_det_train_forward(x, w, cu, bias=b)), (
        f"y differs from the deterministic package ({label})"
    )
    dx_d, dw_d, db_d = _det_backward(det, x, w, cu, dy, b, policy, **outs())
    assert torch.equal(dx, dx_d), f"dx differs from the deterministic package ({label})"
    for got, expected, name in ((dw, dw_d, "dweight"), (db, db_d, "dbias")):
        if got is None:
            continue
        assert got.dtype == expected.dtype, f"{name} {label}"
        atol, rtol = _grad_tolerance(policy, got.dtype)
        _assert_within(
            got.float(),
            expected.float(),
            atol=atol,
            rtol=rtol,
            label=f"{name} vs det {label}",
        )


@pytest.mark.parametrize("policy", ("bf16", "fp32"))
@pytest.mark.parametrize("case", ROUTE_CASES, ids=_case_id)
def test_bf16_dweight_out_with_f32_taps_keeps_dx_bitwise(case, policy) -> None:
    """FP32 taps with a BF16 ``dweight`` out (``bwd_dt`` under the default policy, the fp32g route's BF16 store
    under ``"fp32"``) give the FP32-out call's ``dx`` bits: the tap load form is the same, only the store differs;
    the BF16 ``dweight`` is the FP32 one's within one BF16 rounding."""

    _require_route(
        _combo_route(F32, None, policy), _combo_route(F32, None, policy, BF16)
    )
    S, C, docs = case
    x, w, cu, dy = _inputs(S, C, docs, key=7)
    dx, dw, _ = _backward(x, w, cu, dy, None, policy)
    dx_dt, dw_dt, _ = _backward(
        x,
        w,
        cu,
        dy,
        None,
        policy,
        dweight=torch.empty(C, WIDTH, dtype=BF16, device=_device()),
    )
    assert dw.dtype == F32 and dw_dt.dtype == BF16
    assert torch.equal(dx_dt, dx), (
        f"BF16-out dx differs from the FP32-out dx at {case} {policy}"
    )
    atol, rtol = _grad_tolerance(policy, BF16)
    _assert_within(
        dw_dt.float(),
        dw,
        atol=atol,
        rtol=rtol,
        label=f"bf16 dweight vs the f32 one {case} {policy}",
    )


@pytest.mark.parametrize("kind", tuple(ROUTE_SETTINGS))
def test_backward_programs_switch_on_channel_count(kind, monkeypatch) -> None:
    """Exactly one program of the route runs per call and every program the route ships on this architecture runs
    for some ``(S, C)``: each matches the reference on both sides of its channel threshold (and of its token floor,
    where its rule has one) on a caller-owned workspace that reads back zero, with ``dx`` bitwise the deterministic
    package's (when importable)."""

    _require_route(kind)
    arch, device = _arch(), _device()
    names = loader.route_modules(arch, kind)
    thresholds = _backward_thresholds(arch, kind)
    channels = sorted({C for T in thresholds for C in (T - 8, T, T + 8)} | {8192})
    rows = [
        (S, docs, C) for S, docs in ((5, (2, 0, 3)), (1000, "perf")) for C in channels
    ]
    # A token-floored program runs from its floor only: both sides of the floor at its channel threshold.
    for T, floor in _token_floors(arch, kind).values():
        rows += [(S, "perf", C) for S in (floor - 8, floor) for C in (T - 8, T, T + 8)]
    det = _det_or_none()
    seen = []
    launch = pkg.run_module

    def record_launch(name, values, *, device_index, grid=None):
        seen.append(name)
        return launch(name, values, device_index=device_index, grid=grid)

    monkeypatch.setattr(pkg, "run_module", record_launch)
    launched = set()
    for S, docs, C in rows:
        (selected,) = loader.selected_modules(arch, kind, C=C, S=S)
        label = f"{kind} S={S} C={C} docs={docs} program={selected}"
        x, w, cu, dy, b, policy = _route_inputs(S, C, docs, kind, key=6)
        workspace = pkg.cake_causal_conv1d_train_backward_workspace(
            S, C, device, deterministic=False
        )
        seen.clear()
        dx, dw, db = _backward(x, w, cu, dy, b, policy, workspace=workspace)
        assert seen == [selected], f"{label}: launched {seen}"
        launched.add(selected)
        _check_grads(
            dx, dw, db, reference_backward(x, w, cu, dy, b=b, policy=policy),
            policy=policy, label=label,
        )  # fmt: skip
        assert int(torch.count_nonzero(workspace)) == 0, label
        if det is not None:
            dx_d, _, _ = _det_backward(det, x, w, cu, dy, b, policy)
            assert torch.equal(dx, dx_d), (
                f"dx differs from the deterministic package ({label})"
            )
    assert launched == set(names), f"never launched {set(names) - launched}"


@pytest.mark.parametrize("kind", tuple(ROUTE_SETTINGS))
def test_large_case_over_two_to_the_31_elements(kind) -> None:
    """``S x C > 2^31`` elements (64-bit offsets) at ``C`` 8192: ``y``, ``dx``, ``dweight`` and ``dbias`` against
    the reference, ``dx`` bitwise the deterministic package's (when importable), the workspace left zero."""

    _require_route(kind)
    free, _total = torch.cuda.mem_get_info(_device())
    if free < LARGE_CASE_FREE_BYTES:
        pytest.skip(
            f"needs {LARGE_CASE_FREE_BYTES >> 30} GiB free, {free >> 30} GiB available"
        )
    S, C, docs = LARGE_CASE
    assert S * C > 2**31
    device = _device()
    label = f"{kind} {_case_id(LARGE_CASE)}"
    x, w, cu, dy, b, policy = _route_inputs(S, C, docs, kind, key=5)
    y = _forward(x, w, cu, bias=b)
    _assert_within(
        y, reference_forward(x, w, cu, b), atol=Y_ATOL, rtol=Y_RTOL, label=f"y {label}"
    )
    del y
    workspace = pkg.cake_causal_conv1d_train_backward_workspace(
        S, C, device, deterministic=False
    )
    dx, dw, db = _backward(x, w, cu, dy, b, policy, workspace=workspace)
    assert int(torch.count_nonzero(workspace)) == 0, label
    _check_grads(
        dx, dw, db, reference_backward(x, w, cu, dy, b=b, policy=policy),
        policy=policy, label=label,
    )  # fmt: skip
    det = _det_or_none()
    if det is not None:
        dx_d, _, _ = _det_backward(det, x, w, cu, dy, b, policy)
        assert torch.equal(dx, dx_d), (
            f"dx differs from the deterministic package ({label})"
        )
        del dx_d
    del x, dy, dx
    torch.cuda.empty_cache()


@pytest.mark.parametrize("kind", tuple(ROUTE_SETTINGS))
@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_matches_the_deterministic_package(case, kind, record_property) -> None:
    """``y`` and ``dx`` are bitwise the deterministic package's (the same programs / per-thread arithmetic);
    ``dweight`` / ``dbias`` are within the route's budget of the deterministic ones."""

    det = _det()
    _require_route(kind)
    S, C, docs = case
    x, w, cu, dy, b, policy = _route_inputs(S, C, docs, kind, key=2)
    y = _forward(x, w, cu, bias=b)
    assert torch.equal(y, det.cake_causal_conv1d_det_train_forward(x, w, cu, bias=b)), (
        f"y differs from the deterministic package ({kind} {case})"
    )
    dx, dw, db = _backward(x, w, cu, dy, b, policy)
    dx_d, dw_d, db_d = _det_backward(det, x, w, cu, dy, b, policy)
    assert torch.equal(dx, dx_d), (
        f"dx differs from the deterministic package ({kind} {case})"
    )
    atol, rtol = _grad_tolerance(policy, dw.dtype)
    worst = _assert_within(
        dw.float(),
        dw_d.float(),
        atol=atol,
        rtol=rtol,
        label=f"dweight vs det {kind} {case}",
    )
    record_property(
        "dweight_vs_det_max_abs", float((dw.double() - dw_d.double()).abs().max())
    )
    record_property("dweight_vs_det_worst_ratio_to_budget", worst)
    if db is not None:
        atol, rtol = _grad_tolerance(policy, db.dtype)
        _assert_within(
            db.float(),
            db_d.float(),
            atol=atol,
            rtol=rtol,
            label=f"dbias vs det {kind} {case}",
        )


@pytest.mark.parametrize("kind", tuple(ROUTE_SETTINGS))
@pytest.mark.parametrize("case", REPEAT_CASES, ids=_case_id)
def test_repeated_launches_worst_case(case, kind, record_property) -> None:
    """``REPEATS`` launches on the same inputs: ``dx`` bitwise equal every time (and to the deterministic
    package's when importable), every ``dweight`` / ``dbias`` within the route's budget of the reference; the
    worst case over the repeats and the number of distinct ``dweight`` bit patterns are recorded (the order of
    the cross-CTA sums may vary; no bound is placed on that count)."""

    _require_route(kind)
    S, C, docs = case
    x, w, cu, dy, b, policy = _route_inputs(S, C, docs, kind, key=3)
    refs = reference_backward(x, w, cu, dy, b=b, policy=policy)
    workspace = pkg.cake_causal_conv1d_train_backward_workspace(
        S, C, _device(), deterministic=False
    )
    dx0 = None
    worst_dw = worst_db = 0.0
    patterns = set()
    for repeat in range(REPEATS):
        dx, dw, db = _backward(x, w, cu, dy, b, policy, workspace=workspace)
        if dx0 is None:
            dx0 = dx.clone()
        else:
            assert torch.equal(dx, dx0), (
                f"dx changed at repeat {repeat} ({kind} {case})"
            )
        r_dw, r_db = _check_grads(
            dx, dw, db, refs, policy=policy, label=f"{kind} {case} repeat {repeat}"
        )
        worst_dw, worst_db = max(worst_dw, r_dw), max(worst_db, r_db)
        patterns.add(bytes(dw.contiguous().view(torch.uint8).cpu().numpy()))
        assert int(torch.count_nonzero(workspace)) == 0, (
            f"the workspace is not zero after repeat {repeat}"
        )
    print(
        f"{kind} S={S} C={C}: {REPEATS} launches, worst dweight |err|/budget {worst_dw:.3f}, "
        f"dbias {worst_db:.3f}, {len(patterns)} distinct dweight bit patterns"
    )
    record_property("repeats", REPEATS)
    record_property("worst_dweight_ratio_to_budget", worst_dw)
    record_property("worst_dbias_ratio_to_budget", worst_db)
    record_property("distinct_dweight_patterns", len(patterns))
    try:
        det = cake_package("cake_causal_conv1d_det_train")
    except Exception:  # noqa: BLE001 - parity only where the det package is importable
        return
    dx_d, _, _ = _det_backward(det, x, w, cu, dy, b, policy)
    assert torch.equal(dx0, dx_d), (
        f"dx differs from the deterministic package ({kind} {case})"
    )


def _rel_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float(
        torch.linalg.vector_norm(actual.double() - expected.double())
        / torch.linalg.vector_norm(expected.double())
    )


@pytest.mark.parametrize("case", PRECISION_CASES, ids=_case_id)
def test_fp32_grad_perf_size_precision(case, record_property) -> None:
    """The fp32 policy's perf-size gate with FP64 accumulators: the worst rel-L2 of ``dweight`` against the FP64
    reference over ``PRECISION_REPEATS`` launches is at most ``FP32G_REL_L2_GATE`` (the FP64 final reduction's
    accuracy is kept), and below the default policy's."""

    _require_route("bwd_fp32g")
    S, C, docs = case
    x, w, cu, dy = _inputs(S, C, docs)
    _dx, dw_ref, _ = reference_backward(x, w, cu, dy, policy="fp32")
    worst = 0.0
    for _ in range(PRECISION_REPEATS):
        _dx, dw, _ = _backward(x, w, cu, dy, None, "fp32")
        worst = max(worst, _rel_l2(dw, dw_ref))
    _dx, dw_default, _ = _backward(x, w, cu, dy, None, "bf16")
    rel_l2_default = _rel_l2(dw_default, dw_ref)
    print(
        f"S={S} C={C}: worst dweight rel-L2 vs FP64 over {PRECISION_REPEATS} launches -- fp32 policy {worst:.3e} "
        f"(gate {FP32G_REL_L2_GATE:.1e}), bf16 policy {rel_l2_default:.3e}"
    )
    record_property("dweight_rel_l2_fp32_worst", worst)
    record_property("dweight_rel_l2_bf16", rel_l2_default)
    assert worst <= FP32G_REL_L2_GATE
    assert worst < rel_l2_default
    torch.cuda.empty_cache()


def test_document_isolation() -> None:
    """Perturbing one document (the 3-token document, rows 1000..1002) changes no output or ``dx`` row of another
    document."""

    S, C, docs = 4097, 512, (1000, 0, 3, 3094)
    x, w, cu, dy = _inputs(S, C, docs, key=5)
    y, (dx, _, _) = _forward(x, w, cu), _backward(x, w, cu, dy)
    x2 = x.clone()
    x2[1000:1003] = torch.randn(3, C, device=_device()).to(BF16)
    y2, (dx2, _, _) = _forward(x2, w, cu), _backward(x2, w, cu, dy)
    others = torch.cat([torch.arange(0, 1000), torch.arange(1003, S)]).to(_device())
    assert torch.equal(y[others], y2[others]) and torch.equal(dx[others], dx2[others])


def test_batched_layout_matches_the_packed_one() -> None:
    S, C, docs = 1000, 512, "perf"
    x, w, cu, dy = _inputs(S, C, docs)
    y = _forward(x, w, cu)
    dx, dw, _ = _backward(x, w, cu, dy)
    y3 = _forward(x.unsqueeze(0), w, cu)
    dx3, dw3, _ = _backward(x.unsqueeze(0), w, cu, dy.unsqueeze(0))
    assert tuple(y3.shape) == (1, S, C) and torch.equal(y3[0], y)
    assert torch.equal(dx3[0], dx)
    _assert_within(dw3, dw, atol=DW_ATOL, rtol=DW_RTOL, label="[1, S, C] dweight")


# ----------------------------------------------------------------------------------------------------------------
# The workspace contract.


def test_workspace_bytes_depend_on_channels_only() -> None:
    device = _device()
    arch = _arch()
    for C in (128, 512, 1160, 1280, 1344, 4096, 8192):
        nbytes = pkg.cake_causal_conv1d_train_backward_workspace_bytes(
            1, C, device, deterministic=False
        )
        assert nbytes == pkg.cake_causal_conv1d_train_backward_workspace_bytes(
            10**6, C, device, deterministic=False
        )
        assert nbytes == loader.backward_workspace_bytes(arch, C)
        assert nbytes % loader.WORKSPACE_ALIGN == 0
        # The largest owner holds FP64 bias accumulators (40 bytes per channel) where the architecture has them.
        assert nbytes >= 4 * WIDTH * C
        ws = pkg.cake_causal_conv1d_train_backward_workspace(
            1, C, device, deterministic=False
        )
        assert ws.dtype == torch.uint8 and ws.numel() == nbytes
        assert int(torch.count_nonzero(ws)) == 0


@pytest.mark.parametrize("kind", tuple(ROUTE_SETTINGS))
def test_changed_shapes_reuse_one_zero_workspace(kind) -> None:
    """One caller-owned workspace, sized for the largest ``C``, serves a sequence of shapes of the route (and the
    package's own stream workspace serves the same sequence): every result is within budget and the buffer reads
    back all zero after every launch."""

    _require_route(kind)
    device = _device()
    shapes = ((4097, 1280, (1000, 0, 3, 3094)), (65, 256, (64, 1)), (1000, 1344, "perf"),
              (5, 136, (2, 0, 3)), (4097, 1160, (1000, 0, 3, 3094)), (1000, 512, "perf"))  # fmt: skip
    workspace = pkg.cake_causal_conv1d_train_backward_workspace(
        1, max(C for _, C, _ in shapes), device, deterministic=False
    )
    for own in (True, False):
        for S, C, docs in shapes:
            x, w, cu, dy, b, policy = _route_inputs(S, C, docs, kind, key=6)
            refs = reference_backward(x, w, cu, dy, b=b, policy=policy)
            kwargs = {"workspace": workspace} if own else {}
            dx, dw, db = _backward(x, w, cu, dy, b, policy, **kwargs)
            _check_grads(
                dx, dw, db, refs, policy=policy, label=f"{kind} S={S} C={C} own={own}"
            )
            held = workspace if own else _stream_workspace(device)
            assert held is not None
            assert int(torch.count_nonzero(held)) == 0, (
                f"workspace not zero after {kind} S={S} C={C} (caller-owned: {own})"
            )


@pytest.mark.parametrize("kind", tuple(ROUTE_SETTINGS))
def test_nan_poisoned_outputs_are_overwritten(kind) -> None:
    _require_route(kind)
    for S, C, docs in ROUTE_CASES:
        x, w, cu, dy, b, policy = _route_inputs(S, C, docs, kind, key=7)
        dx = torch.full_like(x, float("nan"))
        dw = torch.full((C, WIDTH), float("nan"), dtype=w.dtype, device=_device())
        db = (
            None
            if b is None
            else torch.full((C,), float("nan"), dtype=b.dtype, device=_device())
        )
        y = torch.full_like(x, float("nan"))
        _forward(x, w, cu, bias=b, out=y)
        got = _backward(x, w, cu, dy, b, policy, dx=dx, dweight=dw, dbias=db)
        assert got[0] is dx and got[1] is dw and got[2] is db
        assert torch.isfinite(y.float()).all()
        _check_grads(
            dx, dw, db, reference_backward(x, w, cu, dy, b=b, policy=policy),
            policy=policy, label=f"NaN-poisoned {kind} S={S} C={C}",
        )  # fmt: skip


def test_workspace_validation() -> None:
    S, C = 64, 256
    x, w, cu, dy = _inputs(S, C, None)
    nbytes = pkg.cake_causal_conv1d_train_backward_workspace_bytes(
        S, C, _device(), deterministic=False
    )
    with pytest.raises(ValueError, match="workspace holds"):
        _backward(
            x, w, cu, dy, workspace=torch.zeros(16, dtype=torch.uint8, device=_device())
        )
    with pytest.raises(ValueError, match="uint8"):
        _backward(
            x,
            w,
            cu,
            dy,
            workspace=torch.zeros(nbytes // 4, dtype=torch.int32, device=_device()),
        )
    with pytest.raises(ValueError, match="aligned"):
        big = torch.zeros(nbytes + 16, dtype=torch.uint8, device=_device())
        _backward(x, w, cu, dy, workspace=big[1 : nbytes + 1])
    with pytest.raises(ValueError, match="1-D"):
        _backward(
            x,
            w,
            cu,
            dy,
            workspace=torch.zeros(1, nbytes, dtype=torch.uint8, device=_device()),
        )


def test_workspace_views_are_built_once_per_workspace() -> None:
    """The ``acc`` / ``work`` tensors of a workspace are built once per layout and reused by later calls; they
    alias its storage but not the tensor, follow the storage when it moves, and are dropped with the
    workspace."""

    S, C, docs = 1000, 1280, "perf"
    device = _device()
    x, w, cu, dy = _inputs(S, C, docs)
    refs = reference_backward(x, w, cu, dy)
    nbytes = pkg.cake_causal_conv1d_train_backward_workspace_bytes(
        S, C, device, deterministic=False
    )
    workspace = pkg.cake_causal_conv1d_train_backward_workspace(
        S, C, device, deterministic=False
    )
    entries = len(pkg._ACCUMULATOR_VIEWS)
    for call in range(2):
        dx, dw, _ = _backward(x, w, cu, dy, workspace=workspace)
        _check_grads(dx, dw, None, refs, policy="bf16", label=f"call {call}")
    held = pkg._ACCUMULATOR_VIEWS.get(workspace)
    assert held is not None and len(held) == 1
    ((ptr, acc, work),) = held.values()
    assert ptr == workspace.data_ptr()
    assert acc._base is None and work._base is None
    storage = workspace.untyped_storage().data_ptr()
    assert acc.untyped_storage().data_ptr() == storage
    assert work.untyped_storage().data_ptr() == storage
    moved = torch.zeros(nbytes, dtype=torch.uint8, device=device)
    workspace.set_(moved.untyped_storage(), 0, (nbytes,), (1,))
    del moved
    dx, dw, _ = _backward(x, w, cu, dy, workspace=workspace)
    _check_grads(dx, dw, None, refs, policy="bf16", label="moved storage")
    ((ptr, acc, work),) = pkg._ACCUMULATOR_VIEWS.get(workspace).values()
    assert ptr == workspace.data_ptr()
    assert acc.untyped_storage().data_ptr() == workspace.untyped_storage().data_ptr()
    assert int(torch.count_nonzero(workspace)) == 0
    del workspace, held, acc, work
    gc.collect()
    assert len(pkg._ACCUMULATOR_VIEWS) == entries


# ----------------------------------------------------------------------------------------------------------------
# Graphs, streams, launches.


def _capture_and_replay(cases, kind: str, *, key: int, own_workspace: bool) -> int:
    """One forward + backward graph per shape, replayed ``GRAPH_REPLAYS`` times into NaN-filled outputs: ``y`` and
    ``dx`` equal eager bitwise, the gradients are within budget of the reference, and a caller-owned workspace
    reads back all zero after every replay."""

    device = _device()
    captured = []
    for S, C, docs in cases:
        x, w, cu, dy, b, policy = _route_inputs(S, C, docs, kind, key=key)
        eager_y = _forward(x, w, cu, bias=b)
        eager_dx, _, _ = _backward(x, w, cu, dy, b, policy)
        refs = reference_backward(x, w, cu, dy, b=b, policy=policy)
        y = torch.empty_like(x)
        dx = torch.empty_like(x)
        dw = torch.empty(C, WIDTH, dtype=w.dtype, device=device)
        db = None if b is None else torch.empty(C, dtype=b.dtype, device=device)
        workspace = (
            pkg.cake_causal_conv1d_train_backward_workspace(
                S, C, device, deterministic=False
            )
            if own_workspace
            else None
        )

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
            _forward(x, w, cu, bias=b, out=y)
            _backward(
                x, w, cu, dy, b, policy, workspace=ws, dx=dx, dweight=dw, dbias=db
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
        captured.append(
            (
                graph,
                (x, w, cu, dy, b),
                (y, dx, dw, db),
                (eager_y, eager_dx),
                refs,
                policy,
                workspace,
            )
        )
    replays = 0
    for graph, _alive, (y, dx, dw, db), (
        eager_y,
        eager_dx,
    ), refs, policy, workspace in captured:
        for _replay in range(GRAPH_REPLAYS):
            for t in (y, dx, dw, db):
                if t is not None:
                    t.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize(device)
            replays += 1
            assert torch.equal(y, eager_y), (
                f"y of replay {replays} differs from eager ({kind})"
            )
            assert torch.equal(dx, eager_dx), (
                f"dx of replay {replays} differs from eager ({kind})"
            )
            _check_grads(
                dx, dw, db, refs, policy=policy, label=f"replay {replays} ({kind})"
            )
            if workspace is not None:
                assert int(torch.count_nonzero(workspace)) == 0
    return replays


@pytest.mark.parametrize("own_workspace", (True, False), ids=("own-ws", "pkg-ws"))
@pytest.mark.parametrize("kind", tuple(ROUTE_SETTINGS))
def test_cuda_graph_replays(kind, own_workspace) -> None:
    _require_route(kind)
    cases = GRAPH_CASES if kind == "bwd" else ROUTE_CASES[:3]
    replays = _capture_and_replay(cases, kind, key=11, own_workspace=own_workspace)
    assert replays == GRAPH_REPLAYS * len(cases)


def test_two_streams_with_two_workspaces() -> None:
    """Two backward calls running at once on two streams, each with its own workspace (caller-owned, then the
    package's per-stream buffers): both match the single-stream result (``dx`` bitwise, gradients within
    budget) and every workspace is zero afterwards."""

    device = _device()
    S, C, docs = 65536, 1280, "perf"
    problems = [_inputs(S, C, docs, key=k) for k in (21, 22)]
    expected = [_backward(x, w, cu, dy) for x, w, cu, dy in problems]
    refs = [reference_backward(x, w, cu, dy) for x, w, cu, dy in problems]
    streams = [torch.cuda.Stream(device=device) for _ in problems]
    for own in (True, False):
        workspaces = [
            pkg.cake_causal_conv1d_train_backward_workspace(
                S, C, device, deterministic=False
            )
            if own
            else None
            for _ in problems
        ]
        results = [None, None]
        torch.cuda.synchronize(device)
        for i, ((x, w, cu, dy), stream) in enumerate(
            zip(problems, streams, strict=True)
        ):
            with torch.cuda.stream(stream):
                kwargs = {"workspace": workspaces[i]} if own else {}
                results[i] = _backward(x, w, cu, dy, **kwargs)
        torch.cuda.synchronize(device)
        for i, stream in enumerate(streams):
            dx, dw, _ = results[i]
            assert torch.equal(dx, expected[i][0]), f"dx of stream {i} (own={own})"
            _check_grads(
                dx, dw, None, refs[i], policy="bf16", label=f"stream {i} own={own}"
            )
            if own:
                assert int(torch.count_nonzero(workspaces[i])) == 0
            else:
                with torch.cuda.stream(stream):
                    held = _stream_workspace(device)
                assert held is not None and int(torch.count_nonzero(held)) == 0


def test_backward_is_one_launch_with_zeroed_accumulators(monkeypatch) -> None:
    """A backward call is ONE ordinary launch of the program ``C`` selects (no reduce program): its arguments
    carry the ``acc`` / ``work`` views of the record's accumulator layout (FP32 or FP64 ``slots x taps x C`` words
    as the rule names them, uint32 ``n_ch_tiles`` counters), zero at launch, and the grid of the call's plan, at
    least one CTA per channel tile."""

    S, C, docs = 1000, 1280, "perf"
    device = _device()
    launched = []

    def record_launch(name, values, *, device_index, grid=None):
        launched.append((name, values, grid))
        acc, work = values["acc"], values["work"]
        launched[-1] += (int(torch.count_nonzero(acc)), int(torch.count_nonzero(work)))

    monkeypatch.setattr(pkg, "run_module", record_launch)
    arch = _arch()
    for kind in ROUTE_SETTINGS:
        if _route_pending(arch, kind):
            continue
        launched.clear()
        x, w, cu, dy, b, policy = _route_inputs(S, C, docs, kind)
        _backward(x, w, cu, dy, b, policy)
        assert len(launched) == 1, f"{kind}: {len(launched)} launches"
        name, values, grid, acc_nonzero, work_nonzero = launched[0]
        assert name == loader.selected_modules(arch, kind, C=C, S=S)[0]
        record = loader.MODULES[name]
        assert "reduce" not in record and record["launch"]["cooperative"] is False
        assert grid == loader.launch_grid(record, device_index=device.index, C=C, S=S)
        assert grid[0] >= values["n_ch_tiles"]
        layout = loader.accumulator_layout(record, C, values["n_ch_tiles"])
        acc_dtype = torch.float64 if "fp32g" in kind else torch.float32
        taps = 5 if "bias" in kind else 4
        words = record["workspace_rule"]["slots"] * taps
        assert loader.accumulator_words_per_channel(record) == words
        assert values["acc"].dtype == acc_dtype and values["acc"].numel() == words * C
        assert (
            values["work"].dtype == torch.uint32
            and values["work"].numel() == values["n_ch_tiles"]
        )
        assert layout["acc_elems"] == words * C
        assert acc_nonzero == 0 and work_nonzero == 0
        assert {"x", "w", "cu", "dy", "dx", "dw", "acc", "work"} <= set(values)


@pytest.mark.parametrize("kind", ("bwd_fp32g", "bwd_bias_fp32g"))
@pytest.mark.parametrize("C", (512, 8192))
def test_fp32g_accumulators_follow_the_rendered_rule(kind, C, monkeypatch) -> None:
    """An fp32g backward at ``C`` 512 and 8192 hands its program an FP64 ``acc`` of the size the rendered rule names
    (``slots x taps x C``, however many accumulator copies the program keeps) inside the public workspace; the
    results stay within budget of the reference, ``dx`` is the deterministic package's, and the workspace reads
    back zero."""

    _require_route(kind)
    det = _det()
    S, docs = 4097, (1000, 0, 3, 3094)
    device = _device()
    x, w, cu, dy, b, policy = _route_inputs(S, C, docs, kind, key=11)
    workspace = pkg.cake_causal_conv1d_train_backward_workspace(
        S, C, device, deterministic=False
    )
    seen = []
    launch = pkg.run_module

    def record_launch(name, values, *, device_index, grid=None):
        seen.append(
            (name, values["acc"].dtype, values["acc"].numel(), values["n_ch_tiles"])
        )
        return launch(name, values, device_index=device_index, grid=grid)

    monkeypatch.setattr(pkg, "run_module", record_launch)
    dx, dw, db = _backward(x, w, cu, dy, b, policy, workspace=workspace)
    monkeypatch.undo()
    ((name, acc_dtype, acc_numel, n_ch_tiles),) = seen
    record = loader.MODULES[name]
    layout = loader.accumulator_layout(record, C, n_ch_tiles)
    assert acc_dtype == torch.float64
    assert (
        acc_numel
        == layout["acc_elems"]
        == loader.accumulator_words_per_channel(record) * C
    )
    assert workspace.numel() >= layout["total_bytes"]
    _check_grads(
        dx, dw, db, reference_backward(x, w, cu, dy, b=b, policy=policy),
        policy=policy, label=f"{kind} C={C}",
    )  # fmt: skip
    dx_d, _, _ = _det_backward(det, x, w, cu, dy, b, policy)
    assert torch.equal(dx, dx_d), (
        f"dx differs from the deterministic package ({kind} C={C})"
    )
    assert int(torch.count_nonzero(workspace)) == 0


def test_backward_keeps_its_speed_beside_a_resident_side_kernel(
    record_property,
) -> None:
    """FlashInfer issues #7038 / #7548: one 1-thread kernel resident on another stream must not stall the
    one-launch backward (``busy <= 2 x alone``, medians of 7 after 3 warm-ups, CUDA events on the current stream);
    ``dx`` is bitwise unchanged beside it and ``dweight`` stays within budget."""

    device = _device()
    S, C, docs = 261393, 512, "perf"
    x, w, cu, dy = _inputs(S, C, docs)
    workspace = pkg.cake_causal_conv1d_train_backward_workspace(
        S, C, device, deterministic=False
    )
    dx = torch.empty_like(x)
    dw = torch.empty(C, WIDTH, dtype=torch.float32, device=device)
    side = torch.cuda.Stream(device=device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    def timed(with_side_kernel: bool) -> float:
        torch.cuda.synchronize(device)
        if with_side_kernel:
            with torch.cuda.stream(side):
                torch.cuda._sleep(10_000_000)
        torch.cuda._sleep(1_000_000)
        start.record()
        _backward(x, w, cu, dy, workspace=workspace, dx=dx, dweight=dw)
        end.record()
        torch.cuda.synchronize(device)
        return start.elapsed_time(end)

    for _ in range(3):
        timed(False)
    alone = statistics.median(timed(False) for _ in range(7))
    dx_alone, dw_alone = dx.clone(), dw.clone()
    busy = statistics.median(timed(True) for _ in range(7))
    print(
        f"backward alone {alone:.3f} ms, beside a resident kernel {busy:.3f} ms ({busy / alone:.2f}x)"
    )
    record_property("backward_alone_ms", f"{alone:.4f}")
    record_property("backward_beside_side_kernel_ms", f"{busy:.4f}")
    record_property("busy_over_alone", f"{busy / alone:.3f}")
    assert torch.equal(dx, dx_alone), "dx changed beside the side kernel"
    _assert_within(
        dw, dw_alone, atol=DW_ATOL, rtol=DW_RTOL, label="dweight beside the side kernel"
    )
    assert busy <= 2.0 * alone, (
        f"the backward stalled behind a kernel resident on another stream: {busy:.3f} ms vs {alone:.3f} ms"
    )


def test_zero_tokens_return_empty_outputs() -> None:
    C = 256
    x = torch.empty(0, C, dtype=BF16, device=_device())
    w = torch.zeros(C, WIDTH, dtype=F32, device=_device())
    b = torch.zeros(C, dtype=BF16, device=_device())
    cu = torch.zeros(2, dtype=torch.int32, device=_device())
    assert tuple(_forward(x, w, cu).shape) == (0, C)
    dx, dw, _ = _backward(x, w, cu, x)
    assert tuple(dx.shape) == (0, C) and torch.equal(dw, torch.zeros_like(dw))
    dw_out = torch.full((C, WIDTH), float("nan"), dtype=F32, device=_device())
    db_out = torch.full((C,), float("nan"), dtype=BF16, device=_device())
    dx, dw, db = _backward(x, w, cu, x, b, "fp32", dweight=dw_out, dbias=db_out)
    assert dw is dw_out and db is db_out
    assert torch.equal(dw, torch.zeros_like(dw)) and torch.equal(
        db, torch.zeros_like(db)
    )


# ----------------------------------------------------------------------------------------------------------------
# Autograd, checkpointing, accumulation.


@pytest.mark.parametrize("combo", AUTOGRAD_COMBOS, ids=_combo_id)
def test_autograd_end_to_end(combo) -> None:
    """``cake_causal_conv1d(..., deterministic=False)`` with FP32 / BF16 taps, no / FP32 / BF16 bias, under both
    SiLU' policies: the forward's bits, ``dx`` bitwise the direct backward's, the parameter gradients within
    budget of the reference, in the parameters' dtypes; a non-contiguous incoming gradient is materialized."""

    weight_dtype, bias_dtype, policy = combo
    kind = _combo_route(weight_dtype, bias_dtype, policy)
    _require_route(kind)
    S, C, docs = 1000, 512, "perf"
    x0, w0, cu, dy = _inputs(S, C, docs, key=3, weight_dtype=weight_dtype)
    b0 = None if bias_dtype is None else _bias_input(C, bias_dtype, key=3)
    refs = reference_backward(x0, w0, cu, dy, b=b0, policy=policy)
    dx_direct, _, _ = _backward(x0, w0, cu, dy, b0, policy)
    for incoming in (dy, dy.t().contiguous().t()):
        x = x0.detach().clone().requires_grad_(True)
        w = w0.detach().clone().requires_grad_(True)
        b = None if b0 is None else b0.detach().clone().requires_grad_(True)
        y = pkg.cake_causal_conv1d(
            x, w, cu, b, silu_grad_preactivation=policy, deterministic=False
        )
        assert torch.equal(y.detach(), _forward(x0, w0, cu, bias=b0))
        y.backward(incoming)
        assert x.grad.dtype == BF16 and torch.equal(x.grad, dx_direct)
        assert w.grad.dtype == w0.dtype and (b is None or b.grad.dtype == b0.dtype)
        _check_grads(
            x.grad,
            w.grad,
            None if b is None else b.grad,
            refs,
            policy=policy,
            label=f"autograd {kind} {_combo_id(combo)}",
        )


def test_non_reentrant_checkpoint_is_two_forwards_and_a_backward() -> None:
    """``torch.utils.checkpoint(..., use_reentrant=False)`` around the autograd entry (2 x forward + backward):
    ``y`` and ``dx`` equal the unchecked call's bitwise, the gradients are within budget."""

    from torch.utils.checkpoint import checkpoint

    _require_route("bwd_bias")
    S, C, docs = 4097, 1280, (1000, 0, 3, 3094)
    x0, w0, cu, dy, b0, policy = _route_inputs(S, C, docs, "bwd_bias", key=13)
    refs = reference_backward(x0, w0, cu, dy, b=b0, policy=policy)
    results = []
    for ckpt in (False, True):
        x = x0.detach().clone().requires_grad_(True)
        w = w0.detach().clone().requires_grad_(True)
        b = b0.detach().clone().requires_grad_(True)

        def layer(x, w, b):
            return pkg.cake_causal_conv1d(
                x, w, cu, b, silu_grad_preactivation=policy, deterministic=False
            )

        y = checkpoint(layer, x, w, b, use_reentrant=False) if ckpt else layer(x, w, b)
        y.backward(dy)
        results.append((y.detach(), x.grad, w.grad, b.grad))
        _check_grads(
            x.grad, w.grad, b.grad, refs, policy=policy, label=f"checkpoint={ckpt}"
        )
    assert torch.equal(results[0][0], results[1][0]) and torch.equal(
        results[0][1], results[1][1]
    )


def test_gradient_accumulation() -> None:
    """Two backward passes accumulate into the parameters' ``.grad`` (autograd's sum of the two gradients) within
    twice the budget of the reference sum."""

    S, C, docs = 4097, 512, (1000, 0, 3, 3094)
    x0, w0, cu, dy = _inputs(S, C, docs, key=17)
    dy2 = torch.randn_like(dy.float()).to(BF16)
    w = w0.detach().clone().requires_grad_(True)
    for g in (dy, dy2):
        pkg.cake_causal_conv1d(x0, w, cu, deterministic=False).backward(g)
    _, dw1, _ = reference_backward(x0, w0, cu, dy)
    _, dw2, _ = reference_backward(x0, w0, cu, dy2)
    _assert_within(
        w.grad,
        dw1 + dw2,
        atol=2 * DW_ATOL,
        rtol=2 * DW_RTOL,
        label="accumulated dweight",
    )


# ----------------------------------------------------------------------------------------------------------------
# The numerical-mode keyword and torch's deterministic mode.


def test_every_entry_requires_deterministic_false() -> None:
    S, C = 64, 256
    x, w, cu, dy = _inputs(S, C, None)
    device = _device()
    calls = {
        "forward": lambda d: pkg.cake_causal_conv1d_train_forward(x, w, cu, **d),
        "backward": lambda d: pkg.cake_causal_conv1d_train_backward(x, w, cu, dy, **d),
        "autograd": lambda d: pkg.cake_causal_conv1d(x, w, cu, **d),
        "function": lambda d: pkg.CakeCausalConv1dFunction.apply(x, w, cu, None, "bf16", d.get("deterministic", True)),
        "bwd_ws": lambda d: pkg.cake_causal_conv1d_train_backward_workspace(S, C, device, **d),
        "bwd_ws_bytes": lambda d: pkg.cake_causal_conv1d_train_backward_workspace_bytes(S, C, device, **d),
        "fwd_ws": lambda d: pkg.cake_causal_conv1d_train_forward_workspace(S, C, device, **d),
        "fwd_ws_bytes": lambda d: pkg.cake_causal_conv1d_train_forward_workspace_bytes(S, C, device, **d),
    }  # fmt: skip
    for name, call in calls.items():
        for kwargs in (
            {},
            {"deterministic": True},
            {"deterministic": 0},
            {"deterministic": None},
        ):
            with pytest.raises(ValueError, match="deterministic=False"):
                call(kwargs)
        assert call({"deterministic": False}) is not None, name


def test_torch_deterministic_mode_alerts_on_the_backward() -> None:
    """Under ``torch.use_deterministic_algorithms(True)`` the backward raises (``warn_only=True``: warns and runs);
    the forward (the deterministic programs' bits) runs without an alert."""

    S, C, docs = 1000, 512, "perf"
    x, w, cu, dy = _inputs(S, C, docs)
    expected_y = _forward(x, w, cu)
    prev, prev_warn = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    try:
        torch.use_deterministic_algorithms(True)
        assert torch.equal(_forward(x, w, cu), expected_y)
        with pytest.raises(RuntimeError, match="cake_causal_conv1d_train_backward"):
            _backward(x, w, cu, dy)
        xg = x.clone().requires_grad_(True)
        y = pkg.cake_causal_conv1d(xg, w, cu, deterministic=False)
        with pytest.raises(RuntimeError, match="cake_causal_conv1d_train_backward"):
            y.backward(dy)
        torch.use_deterministic_algorithms(True, warn_only=True)
        with pytest.warns(UserWarning, match="cake_causal_conv1d_train_backward"):
            dx, dw, _ = _backward(x, w, cu, dy)
        _check_grads(
            dx,
            dw,
            None,
            reference_backward(x, w, cu, dy),
            policy="bf16",
            label="warn_only",
        )
    finally:
        torch.use_deterministic_algorithms(prev, warn_only=prev_warn)


def test_argument_validation() -> None:
    S, C = 64, 256
    x, w, cu, dy = _inputs(S, C, None)
    b = _bias_input(C, F32)
    with pytest.raises(ValueError, match="multiple of 8"):
        _forward(x[:, :130].contiguous(), w[:130].contiguous(), cu)
    with pytest.raises(ValueError, match="width-4"):
        _forward(x, w[:, :3].contiguous(), cu)
    with pytest.raises(ValueError, match="torch.int32"):
        _forward(x, w, cu.long())
    with pytest.raises(ValueError, match="shape"):
        _backward(x, w, cu, dy[: S - 1])
    with pytest.raises(
        ValueError, match="weight must be torch.float32 or torch.bfloat16"
    ):
        _backward(x, w.half(), cu, dy)
    with pytest.raises(ValueError, match="dbias"):
        _backward(x, w, cu, dy, dbias=torch.empty_like(b))
    with pytest.raises(ValueError, match="silu_grad_preactivation"):
        _backward(x, w, cu, dy, policy="fp16")
    with pytest.raises(ValueError, match="CUDA"):
        _forward(x.cpu(), w.cpu(), cu.cpu())


def test_fp32_dweight_must_be_16_byte_aligned() -> None:
    """The backward programs store one channel's four FP32 ``dweight`` taps with one 16-byte store: an FP32
    ``dweight`` out off the 16-byte grid is refused (``ValueError``) before anything launches; a 16-byte aligned
    view inside a larger buffer is written, and a BF16 ``dweight`` or an FP32 / BF16 ``dbias`` (stored per
    element) is accepted at any element offset."""

    S, C, docs = 1000, 512, "perf"
    device = _device()
    x, w, cu, dy = _inputs(S, C, docs, key=8)
    b = _bias_input(C, F32, key=8)
    refs = reference_backward(x, w, cu, dy, b=b)
    n = C * WIDTH
    buf = torch.zeros(n + 4, dtype=F32, device=device)
    workspace = pkg.cake_causal_conv1d_train_backward_workspace(
        S, C, device, deterministic=False
    )
    for bias in (None, b):
        for offset in (1, 2, 3):
            misaligned = buf[offset : offset + n].view(C, WIDTH)
            assert misaligned.data_ptr() % 16 != 0
            with pytest.raises(ValueError, match="16-byte aligned"):
                _backward(x, w, cu, dy, bias, dweight=misaligned, workspace=workspace)
    torch.cuda.synchronize(device)
    assert (
        int(torch.count_nonzero(buf)) == 0 and int(torch.count_nonzero(workspace)) == 0
    )
    aligned = buf[4:].view(C, WIDTH)
    db_view = torch.zeros(C + 1, dtype=F32, device=device)[1:]
    dx, dw, db = _backward(
        x, w, cu, dy, b, dweight=aligned, dbias=db_view, workspace=workspace
    )
    assert dw is aligned and db is db_view
    _check_grads(dx, dw, db, refs, policy="bf16", label="16-byte aligned dweight view")
    for dweight_offset, dbias_offset in ((4, 1), (2, 3)):
        dw16 = torch.zeros(n + 4, dtype=BF16, device=device)[dweight_offset:][:n].view(
            C, WIDTH
        )
        db16 = torch.zeros(C + 4, dtype=BF16, device=device)[dbias_offset:][:C]
        assert dw16.data_ptr() % 16 != 0
        dx, dw, db = _backward(
            x, w, cu, dy, b, dweight=dw16, dbias=db16, workspace=workspace
        )
        assert dw is dw16 and db is db16
        _check_grads(
            dx,
            dw,
            db,
            refs,
            policy="bf16",
            label=f"bf16 outs at element offsets {dweight_offset} / {dbias_offset}",
        )
    assert int(torch.count_nonzero(workspace)) == 0


# ----------------------------------------------------------------------------------------------------------------
# The registry against the rendered manifest.


def test_every_route_ships_on_every_architecture() -> None:
    libraries = cake_build.libraries_by_name()
    for arch in loader.ARCHES:
        assert set(loader.ROUTES[arch]) == set(loader.KINDS), arch
        for kind in loader.KINDS:
            for name in loader.route_modules(arch, kind):
                assert loader.library_name(name, arch) in libraries, (arch, kind, name)


def test_registry_records_reference_the_build_manifest() -> None:
    """Every record names its ``build.json`` library (kernel + binding sources under ``csrc/``, digests) and no
    record is left a TODO-EXPORT placeholder; the manifest names this package's import path."""

    manifest = cake_build.manifest()
    assert manifest["schema"] == "cake-build/1"
    assert (
        manifest["package"]
        == "flashinfer.experimental.cake.cake_causal_conv1d_train_backend"
    )
    libraries = cake_build.libraries_by_name()
    kcp_loader = cake_package(KEY, "kcp_jit")
    referenced = set()
    for registry in (loader.MODULES, kcp_loader.MODULES):
        for name, record in registry.items():
            assert "TODOEXPORT" not in name, f"{name} is still a placeholder"
            assert record["library"] == record["kernel"] == name
            assert record["kernel_symbol"] == f"kernel_{name}"
            for arch in record["arches"]:
                entry = libraries[f"{name}_{arch}"]
                assert entry["sources"] == [
                    f"csrc/{name}_kernel.cu",
                    f"csrc/{name}_binding.cu",
                ]
                referenced.add(entry["name"])
    assert referenced == set(libraries)
