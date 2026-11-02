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

Tests for the Cake fused BF16 RMSNorm training kernels with the
non-deterministic weight gradient (``deterministic=False``; SM100 / SM103 /
SM107).

Every kernel output is compared with an FP64 reference of the same math: BF16
``y`` / ``dx`` at ``atol = rtol = 1e-2`` after normalizing both sides to unit
scale (so 1e-3 and 1e3 magnitude inputs are judged like N(0, 1) inputs), FP32
``rstd`` at relative L2 <= 1e-5 and FP32 ``dw`` at relative L2 <= 1e-4 with
every element within ``atol = rtol = 1e-4`` of the reference -- the
deterministic package's budget.

A backward window runs one of two forms (the registry's records for the call's
token count): the single kernel, which adds the CTAs' ``dw`` sums in FP64 in a
scheduling-dependent order, or the deterministic package's pair rendered into
this package.  Every case derives its form from its own window (the records of
its architecture, width, residual flag, weight dtype and token count, read from
``ROUTES`` and the select rules) and checks that the call launched exactly
those records.  A single-kernel case also holds its ``dw`` error to the eager
FP32 reduction ``(g * x * rstd).sum(0)`` of the same inputs (not worse, up to
one FP32 ulp of the result); a pair case is the deterministic package's ``dw``,
which is judged on the median over the matrix like that package's suite.

The matrix covers token counts from 0 to 65536 (the deterministic contract's
rows, the observed training counts and every program window edge +-1 of the
registry), both eps values where the contract names both, 1e-3 / 1e3 scales,
zero inputs and an extreme row, row-strided inputs, the residual-add fusion
and the FP32-weight routes (also combined), the ``[..., H]`` per-head views,
repeated launches (worst case over 20, ``dx`` bitwise stable) compared with the
deterministic package where it serves the call, a workspace reused across token
counts, windows and weight dtypes with its documented rest state checked after
every call that reuses one, two streams, CUDA Graph capture and replay, the
autograd wrapper (gradient
accumulation, saved tensors, non-reentrant checkpointing, the mode resolved
again when the backward runs) and argument validation.  The build manifest's
per-architecture CUDA floors and flags are checked without a device.

This suite exercises the backend packages alone, through
``tests.test_helpers.cake_package``; the facade and the ``flashinfer`` top
level are ``test_cake_rmsnorm_train_api.py``'s.
"""

from __future__ import annotations

import dataclasses
import functools
import statistics
from pathlib import Path
from typing import Optional

import pytest
import torch
import torch.utils.checkpoint

from tests.test_helpers.cake_package import cake_package

pkg = cake_package("cake_rmsnorm_train")
loader = cake_package("cake_rmsnorm_train", "cake_jit")
cake_build = cake_package("cake_rmsnorm_train", "cake_build")
det = cake_package("cake_rmsnorm_det_train")
det_loader = cake_package("cake_rmsnorm_det_train", "cake_jit")
det_build = cake_package("cake_rmsnorm_det_train", "cake_build")
forward = functools.partial(pkg.cake_rmsnorm_train_forward, deterministic=False)
backward = functools.partial(pkg.cake_rmsnorm_train_backward, deterministic=False)
workspace_for = functools.partial(
    pkg.cake_rmsnorm_train_backward_workspace, deterministic=False
)
rmsnorm = functools.partial(pkg.cake_rmsnorm, deterministic=False)

SUPPORTED_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
WIDTHS = (
    (6144, 1e-5),
    (5120, 1e-6),
    (4096, 1e-5),
    (2048, 1e-6),
    (1536, 1e-6),
    (512, 1e-6),
    (256, 1e-6),
)
HIDDEN_SIZES = tuple(hidden for hidden, _eps in WIDTHS)
EPS = {hidden: eps for hidden, eps in WIDTHS}
# Widths whose contract names both eps values: their contract cases run at each.
EPS_VARIANTS = {4096: (1e-5, 1e-6), 1536: (1e-5, 1e-6)}
# The deterministic package's row counts (tile edges, 2^n +- 1, the two contract
# training shapes) and the wider counts of this suite.
ROWS = (1, 2, 7, 63, 64, 65, 127, 128, 129, 255, 257, 1000, 4096, 16172, 16231, 16385)
LARGE_ROWS = (65536,)
# Further counts of the H=1536 / H=4096 contract and the observed training counts.
CONTRACT_ROWS = {
    4096: (1023, 1025, 4097, 14928, 16263, 16384, 32769, 65536),
    1536: (1023, 1025, 4097, 14928, 15566, 16263, 16384, 32769, 65536),
}
OBSERVED_ROWS = (14928, 15288, 15566, 15590, 16114, 16132, 16164, 16263)
SCALES = (1e-3, 1e3)
W_DTYPES = {"bf16": torch.bfloat16, "f32": torch.float32}
REPEATS = 20
RSTD_REL_L2 = 1e-5
DW_REL_L2 = 1e-4
DW_ATOL = 1e-4
DW_RTOL = 1e-4
# A single-kernel dw may exceed the eager FP32 reduction's error by at most the
# rounding of its own FP32 result (one FP32 ulp, relative).
DW_EAGER_FLOOR = 2.0**-23
ATOL = 1e-2
RTOL = 1e-2
SEED = 8066
# The CUDA toolkit floor of every exported architecture (``build.json``
# ``min_cuda``) and the exact code it is compiled for.
MIN_CUDA = {"sm_100a": "12.8", "sm_103a": "12.9", "sm_107a": "13.4"}
GENCODE = {
    "sm_100a": "-gencode=arch=compute_100a,code=sm_100a",
    "sm_103a": "-gencode=arch=compute_103a,code=sm_103a",
    "sm_107a": "-gencode=arch=compute_107a,code=sm_107a",
}
ACCURACY_FLAGS = ("--prec-div=true", "--prec-sqrt=true", "--ftz=false")


def _device() -> torch.device:
    return torch.device("cuda:0")


def _capability() -> tuple[int, int]:
    return torch.cuda.get_device_capability(_device())


def _supported() -> bool:
    return (
        torch.cuda.is_available()
        and _capability() in SUPPORTED_CAPABILITIES
        and pkg.is_cake_rmsnorm_train_supported(_device(), deterministic=False)
    )


def _arch() -> str:
    return loader.arch_for_capability(_capability())


@pytest.fixture(autouse=True)
def device_or_skip(request):
    """Every test needs a supported device except the ``cpu_only`` ones."""
    if request.node.get_closest_marker("cpu_only") is None and not _supported():
        pytest.skip(
            "cake_rmsnorm_train needs an SM100 / SM103 / SM107 device with exported routes"
        )


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    if _supported():
        for package in (loader, det_loader):
            arch = package.arch_for_capability(_capability())
            for spec in package.specs_for_arch(arch):
                spec.build_and_load()
    yield


@pytest.fixture(autouse=True)
def nondeterministic_allowed():
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(False)
    yield
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


# Every module launch of this package in the running test (name), and the FP32
# ``dw`` buffer each backward module wrote (the kernel's dw before the autograd
# wrapper casts it to the weight dtype).
_LAUNCHED: list[str] = []
_DW_BUFFERS: list[torch.Tensor] = []


@pytest.fixture(autouse=True)
def record_launches(monkeypatch: pytest.MonkeyPatch):
    run_module = loader.run_module

    def recording(name, values, *, rows, device_index):
        _LAUNCHED.append(name)
        if loader.MODULES[name]["role"].startswith("bwd"):
            _DW_BUFFERS.append(values["dw"])
        return run_module(name, values, rows=rows, device_index=device_index)

    monkeypatch.setattr(loader, "run_module", recording)
    _LAUNCHED.clear()
    _DW_BUFFERS.clear()
    yield


# --------------------------------------------------------------------------- the registry, per case


def _w_token(w_dtype: torch.dtype) -> str:
    return "f32" if w_dtype == torch.float32 else "bf16"


def _routes(hidden: int, kind: str, residual: bool, w_dtype: str = "bf16") -> bool:
    return loader.route_applies(
        device_capability=_capability(),
        hidden=hidden,
        kind=kind,
        residual=residual,
        w_dtype=w_dtype,
    )


def _skip_unless_route(hidden: int, residual: bool = False, w_dtype: str = "bf16"):
    if not all(_routes(hidden, kind, residual, w_dtype) for kind in ("fwd", "bwd")):
        pytest.skip(
            f"hidden={hidden} residual={residual} w={w_dtype} has no exported route here"
        )


def _admits(rule: Optional[dict], rows: int) -> bool:
    if rule is None:
        return True
    assert rule["kind"] == "rows_between", rule
    return rows > int(rule["above"]) and (
        rule["at_most"] is None or rows <= int(rule["at_most"])
    )


def _window_records(
    hidden: int, kind: str, residual: bool, w_dtype: str, rows: int
) -> list[str]:
    """The records one call of ``rows`` tokens launches, in route order, read
    from ``ROUTES`` and each record's select rule (independent of the host's
    ``selected_modules``)."""
    key = loader.route_key(hidden, kind, residual, w_dtype)
    return [
        name
        for name in loader.ROUTES[_arch()][key]
        if _admits(loader.MODULES[name]["select_rule"], rows)
    ]


def _form(names: list[str]) -> str:
    """``single`` (one ``claim_sum_f64`` kernel) or ``pair`` (a deterministic
    pair: ``chunk_partials`` main kernel, then its dw reduce) of one backward
    window's records."""
    records = [loader.MODULES[name] for name in names]
    owners = [r["workspace_rule"]["kind"] for r in records if r["workspace_rule"]]
    reduces = [r for r in records if r["role"] == "bwd_reduce"]
    if len(records) == 1 and owners == ["claim_sum_f64"]:
        return "single"
    if len(records) == 2 and owners == ["chunk_partials"] and len(reduces) == 1:
        return "pair"
    raise AssertionError(f"a backward window launches {names}: neither form")


def _run_backward(*args, **kwargs):
    """``((dx, dw), the records this call launched)``."""
    start = len(_LAUNCHED)
    result = backward(*args, **kwargs)
    return result, list(_LAUNCHED[start:])


def _workspace_at_rest(ws: torch.Tensor) -> bool:
    """The documented rest state after every call of any layout: every byte
    before the pair partial rows (``PAIR_ROWS_OFFSET``: the pair counters, the
    single kernel's counters and FP64 sums) is zero, so the buffer serves the
    next call of any layout; the pair partial rows are scratch."""
    torch.cuda.synchronize()
    return not bool(ws[: min(loader.PAIR_ROWS_OFFSET, ws.numel())].any())


def _bytes(rows: int, hidden: int, residual: bool = False) -> int:
    return pkg.cake_rmsnorm_train_backward_workspace_bytes(
        rows, hidden, _device(), residual=residual, deterministic=False
    )


# --------------------------------------------------------------------------- inputs and references


def _generator(*keys) -> torch.Generator:
    seed = SEED
    for key in keys:
        seed = (seed * 1_000_003 + int(key)) % (2**31 - 1)
    return torch.Generator(device=_device()).manual_seed(seed)


def _rows(
    rows: int,
    hidden: int,
    *,
    pad=0,
    scale: float = 1.0,
    zeros: bool = False,
    generator: torch.Generator,
) -> torch.Tensor:
    """BF16 ``[rows, hidden]`` view with ``stride(0) == hidden + pad`` (``pad="2H"``: ``2 * hidden``)."""
    pad = hidden if pad == "2H" else int(pad)
    shape = (max(rows, 1), hidden + pad)
    if zeros:
        base = torch.zeros(shape, dtype=torch.bfloat16, device=_device())
    else:
        base = (
            scale
            * torch.randn(
                shape, dtype=torch.float32, device=_device(), generator=generator
            )
        ).to(torch.bfloat16)
    return base[:rows, :hidden]


def _weight(hidden: int, generator: torch.Generator, dtype=torch.bfloat16):
    w = 1.0 + 0.1 * torch.randn(
        hidden, dtype=torch.float32, device=_device(), generator=generator
    )
    return w.to(dtype)


def _reference_forward(x, w, eps, residual=None):
    """FP64 forward over the last axis; the residual boundary rounds to BF16 like the kernel."""
    normalized = (
        (x.double() + residual.double()).to(torch.bfloat16)
        if residual is not None
        else x
    )
    rstd = torch.rsqrt(normalized.double().pow(2).mean(-1) + eps)
    y = (normalized.double() * rstd.unsqueeze(-1)) * w.double()
    return y, rstd, normalized


def _reference_backward(g, x, w, rstd, g_h=None):
    """FP64 backward at the kernel's FP32 ``rstd``: ``dx`` per row, ``dw`` over every leading axis."""
    xf, wf, gf = x.double(), w.double(), g.double()
    r = rstd.double().unsqueeze(-1)
    a = gf * wf
    c = (a * xf).sum(-1, keepdim=True) / x.shape[-1]
    dx = r * a - xf * r.pow(3) * c
    if g_h is not None:
        dx = dx + g_h.double()
    dw = (gf * xf * r).reshape(-1, x.shape[-1]).sum(0)
    return dx, dw


def _rel_l2(actual: torch.Tensor, reference: torch.Tensor) -> float:
    a, b = actual.double().flatten(), reference.double().flatten()
    den = b.norm().item()
    return (a - b).norm().item() / den if den > 0 else (a - b).norm().item()


def _assert_bf16_close(actual, reference, label: str) -> None:
    assert actual.dtype == torch.bfloat16, label
    a, b = actual.double(), reference.double()
    assert torch.isfinite(a).all(), f"{label} has non-finite values"
    if a.numel() == 0:
        return
    scale = b.pow(2).mean().sqrt().item()
    scale = scale if scale > 0.0 else 1.0
    torch.testing.assert_close(a / scale, b / scale, atol=ATOL, rtol=RTOL, msg=label)


# (label, form, kernel dw rel-L2, eager-FP32-path dw rel-L2) of every dw case checked in this process; the median
# readings per form are the last test of this module.
_DW_ERRORS: list[tuple[str, str, float, float]] = []


def _assert_dw(dw, dw_ref, g, x, rstd, label: str, form: str) -> tuple[float, float]:
    """dw against the FP64 reference at the deterministic budget -- rel-L2 <=
    1e-4 and every element within ``atol = rtol = 1e-4`` -- and, for a
    single-kernel window, not worse than the eager FP32 reduction of the same
    inputs (up to ``DW_EAGER_FLOOR``).  Returns ``(rel-L2, worst element error
    / its tolerance)``."""
    assert dw.dtype == torch.float32 and dw.is_contiguous(), label
    assert torch.isfinite(dw).all(), f"{label} has non-finite values"
    error = _rel_l2(dw, dw_ref)
    assert error <= DW_REL_L2, f"{label}: dw rel-L2 {error:.3e} > {DW_REL_L2:.0e}"
    diff = (dw.double() - dw_ref).abs()
    tolerance = DW_ATOL + DW_RTOL * dw_ref.abs()
    bad = diff > tolerance
    hidden = x.shape[-1]
    eager_fp32 = (
        (g.float() * x.float() * rstd.float().unsqueeze(-1)).reshape(-1, hidden).sum(0)
    )
    eager = _rel_l2(eager_fp32, dw_ref)
    if bool(bad.any()):
        at = int((diff / tolerance).argmax())
        eager_ratio = (eager_fp32.double() - dw_ref).abs() / tolerance
        raise AssertionError(
            f"{label} ({form}): {int(bad.sum())} dw elements outside atol=rtol=1e-4 "
            f"(worst column {at}: {float(dw[at]):.9g} vs {float(dw_ref[at]):.9g}, "
            f"error/tolerance {float(diff[at] / tolerance[at]):.3f}); eager FP32 on the same "
            f"inputs: {int((eager_ratio > 1).sum())} elements outside, worst error/tolerance "
            f"{float(eager_ratio.max()):.3f}, column {at} {float(eager_ratio[at]):.3f}; "
            f"rel-L2 {error:.3e} vs eager {eager:.3e}"
        )
    if form == "single":
        assert error <= max(eager, DW_EAGER_FLOOR), (
            f"{label}: single-kernel dw rel-L2 {error:.3e} > eager FP32 {eager:.3e} "
            f"(ratio {error / eager if eager else float('inf'):.3f})"
        )
    _DW_ERRORS.append((label, form, error, eager))
    return error, float((diff / tolerance).max())


@dataclasses.dataclass
class Case:
    x: torch.Tensor
    u: Optional[torch.Tensor]
    g: torch.Tensor
    g_h: Optional[torch.Tensor]
    w: torch.Tensor
    eps: float
    rstd: torch.Tensor
    normalized: torch.Tensor
    dx: torch.Tensor
    dw: torch.Tensor
    dw_ref: Optional[torch.Tensor]
    records: list[str]
    form: Optional[str]


def _case_inputs(
    hidden: int,
    rows: int,
    *,
    residual=False,
    w_dtype=torch.bfloat16,
    pad=0,
    scale: float = 1.0,
    zeros: bool = False,
    spike: bool = False,
    key: int = 0,
):
    pad_key = hidden if pad == "2H" else int(pad)
    generator = _generator(
        hidden,
        rows,
        residual,
        pad_key,
        w_dtype == torch.float32,
        int(scale * 1e6) % 9973,
        zeros,
        spike,
        key,
    )
    x = _rows(rows, hidden, pad=pad, scale=scale, zeros=zeros, generator=generator)
    if spike and rows > 0:
        row = min(rows - 1, 3)
        x[row] = (x[row].float() * 1e3).to(torch.bfloat16)
        x[row, min(hidden - 1, 17)] = 1e4
    u = (
        _rows(rows, hidden, scale=scale, zeros=zeros, generator=generator)
        if residual
        else None
    )
    g = _rows(rows, hidden, pad=pad, generator=generator)
    g_h = _rows(rows, hidden, generator=generator) if residual else None
    w = _weight(hidden, generator, w_dtype)
    return x, u, g, g_h, w


def _check(
    hidden: int,
    rows: int,
    *,
    residual=False,
    w_dtype=torch.bfloat16,
    pad=0,
    eps: Optional[float] = None,
    scale: float = 1.0,
    zeros: bool = False,
    spike: bool = False,
    key: int = 0,
) -> Case:
    """Forward and backward of one case against FP64; the backward launches
    exactly its window's records, whose form decides the eager gate."""
    eps = EPS[hidden] if eps is None else eps
    x, u, g, g_h, w = _case_inputs(
        hidden,
        rows,
        residual=residual,
        w_dtype=w_dtype,
        pad=pad,
        scale=scale,
        zeros=zeros,
        spike=spike,
        key=key,
    )
    w_token = _w_token(w_dtype)
    start = len(_LAUNCHED)
    outputs = forward(x, w, eps, residual=u)
    fwd_launched = _LAUNCHED[start:]
    y, rstd = outputs[0], outputs[1]
    y_ref, rstd_ref, normalized_ref = _reference_forward(x, w, eps, residual=u)
    label = (
        f"hidden={hidden} rows={rows} residual={residual} w={w_token} pad={pad} "
        f"eps={eps} scale={scale} zeros={zeros} spike={spike}"
    )
    assert tuple(y.shape) == (rows, hidden) and y.is_contiguous()
    assert rstd.dtype == torch.float32 and tuple(rstd.shape) == (rows,)
    _assert_bf16_close(y, y_ref, f"y {label}")
    assert torch.isfinite(rstd).all(), f"rstd {label}"
    assert _rel_l2(rstd, rstd_ref) <= RSTD_REL_L2, f"rstd {label}"
    normalized = outputs[2] if residual else x
    if residual:
        assert torch.equal(normalized, normalized_ref), f"h_new {label}"
    (dx, dw), records = _run_backward(g, normalized, w, rstd, g_residual=g_h)
    dx_ref, dw_ref = _reference_backward(g, normalized, w, rstd, g_h=g_h)
    assert tuple(dx.shape) == (rows, hidden) and dx.is_contiguous()
    _assert_bf16_close(dx, dx_ref, f"dx {label}")
    if rows == 0:
        assert fwd_launched == [] and records == []
        assert torch.equal(dw, torch.zeros_like(dw))
        return Case(x, u, g, g_h, w, eps, rstd, normalized, dx, dw, None, [], None)
    assert fwd_launched == _window_records(hidden, "fwd", residual, w_token, rows), (
        f"forward {label} launched {fwd_launched}"
    )
    expected = _window_records(hidden, "bwd", residual, w_token, rows)
    assert records == expected, (
        f"backward {label} launched {records}, window {expected}"
    )
    form = _form(expected)
    _assert_dw(dw, dw_ref, g, normalized, rstd, f"dw {label}", form)
    return Case(x, u, g, g_h, w, eps, rstd, normalized, dx, dw, dw_ref, records, form)


# --------------------------------------------------------------------------- accuracy matrix


@pytest.mark.parametrize("w", tuple(W_DTYPES))
@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
@pytest.mark.parametrize("rows", ROWS + LARGE_ROWS)
def test_forward_backward_match_fp64(hidden: int, rows: int, w: str) -> None:
    _skip_unless_route(hidden, w_dtype=w)
    _check(hidden, rows, w_dtype=W_DTYPES[w])


def _contract_cases():
    for hidden in HIDDEN_SIZES:
        counts = sorted(set(CONTRACT_ROWS.get(hidden, ())) | set(OBSERVED_ROWS))
        for eps in EPS_VARIANTS.get(hidden, (EPS[hidden],)):
            for w in W_DTYPES if hidden in EPS_VARIANTS else ("bf16",):
                for rows in counts:
                    yield pytest.param(
                        hidden, rows, eps, w, id=f"h{hidden}-T{rows}-eps{eps:g}-{w}"
                    )


@pytest.mark.parametrize(("hidden", "rows", "eps", "w"), list(_contract_cases()))
def test_contract_and_observed_rows(hidden: int, rows: int, eps: float, w: str) -> None:
    """The H=1536 / H=4096 contract rows and the observed training token counts
    at both contract eps values (both weight dtypes); the observed counts at
    every other width."""
    _skip_unless_route(hidden, w_dtype=w)
    _check(hidden, rows, w_dtype=W_DTYPES[w], eps=eps)


@pytest.mark.parametrize(
    ("hidden", "eps"),
    [(hidden, eps) for hidden, variants in EPS_VARIANTS.items() for eps in variants],
)
@pytest.mark.parametrize("rows", (7, 16231))
@pytest.mark.parametrize("scale", (1.0, *SCALES))
@pytest.mark.parametrize("w", tuple(W_DTYPES))
def test_eps_variants(hidden: int, eps: float, rows: int, scale: float, w: str) -> None:
    _skip_unless_route(hidden, w_dtype=w)
    _check(hidden, rows, eps=eps, scale=scale, w_dtype=W_DTYPES[w])


def _route_params():
    for hidden in HIDDEN_SIZES:
        for w in W_DTYPES:
            yield pytest.param(hidden, False, w, id=f"h{hidden}-{w}")
    for w in W_DTYPES:
        yield pytest.param(6144, True, w, id=f"res-h6144-{w}")


def _edge_rows(hidden: int, residual: bool, w: str) -> list[int]:
    """Both neighbours of every program-window edge of the route's forward and
    backward (``program_windows``), and one row."""
    edges = set()
    for kind in ("fwd", "bwd"):
        for above, at_most, _library in loader.program_windows(
            _arch(), hidden, kind, residual, w
        ):
            edges.update(bound for bound in (above, at_most) if bound)
    return sorted({1, *(row for edge in edges for row in (edge - 1, edge, edge + 1))})


@pytest.mark.parametrize(("hidden", "residual", "w"), list(_route_params()))
def test_program_window_edges(hidden: int, residual: bool, w: str) -> None:
    """Every window edge +-1 of the route matches FP64, every call launches its
    window's records, and every forward and backward program of the route runs
    for one of those counts."""
    _skip_unless_route(hidden, residual, w)
    launched = set()
    forms = []
    for rows in _edge_rows(hidden, residual, w):
        case = _check(hidden, rows, residual=residual, w_dtype=W_DTYPES[w], key=5)
        launched.update(case.records)
        launched.update(_window_records(hidden, "fwd", residual, w, rows))
        forms.append(f"{rows}:{case.form}")
    for kind in ("fwd", "bwd"):
        names = set(loader.ROUTES[_arch()][loader.route_key(hidden, kind, residual, w)])
        assert names <= launched, f"{kind} programs never launched: {names - launched}"
    print(f"EDGES h={hidden} res={residual} w={w} {' '.join(forms)}")


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
@pytest.mark.parametrize("pad", (64, "2H"))
@pytest.mark.parametrize("rows", (7, 257, 16385))
@pytest.mark.parametrize("w", tuple(W_DTYPES))
def test_row_strided_inputs(hidden: int, pad, rows: int, w: str) -> None:
    _skip_unless_route(hidden, w_dtype=w)
    _check(hidden, rows, pad=pad, w_dtype=W_DTYPES[w])


@pytest.mark.parametrize("rows", (1, 7, 129, 257, 4096, 16231, 16385, 65536))
@pytest.mark.parametrize("pad", (0, 64))
@pytest.mark.parametrize("w", tuple(W_DTYPES))
def test_residual_fusion(rows: int, pad, w: str) -> None:
    _skip_unless_route(6144, residual=True, w_dtype=w)
    _check(6144, rows, residual=True, pad=pad, w_dtype=W_DTYPES[w])


def _scale_params():
    for hidden in HIDDEN_SIZES:
        for w in W_DTYPES:
            yield pytest.param(hidden, False, w, id=f"h{hidden}-{w}")
    for w in W_DTYPES:
        yield pytest.param(6144, True, w, id=f"res-h6144-{w}")


@pytest.mark.parametrize(("hidden", "residual", "w"), list(_scale_params()))
@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("rows", (7, 16231))
def test_input_scales(hidden: int, residual: bool, w: str, scale: float, rows: int):
    _skip_unless_route(hidden, residual, w)
    _check(hidden, rows, residual=residual, w_dtype=W_DTYPES[w], scale=scale)


@pytest.mark.parametrize(("hidden", "residual", "w"), list(_scale_params()))
@pytest.mark.parametrize("rows", (1, 129, 16231))
def test_zero_input_has_no_nan(hidden: int, residual: bool, w: str, rows: int):
    _skip_unless_route(hidden, residual, w)
    _check(hidden, rows, residual=residual, w_dtype=W_DTYPES[w], zeros=True)


@pytest.mark.parametrize(("hidden", "residual", "w"), list(_scale_params()))
@pytest.mark.parametrize("rows", (257, 16385))
def test_extreme_row(hidden: int, residual: bool, w: str, rows: int) -> None:
    _skip_unless_route(hidden, residual, w)
    _check(hidden, rows, residual=residual, w_dtype=W_DTYPES[w], spike=True)


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
@pytest.mark.parametrize("w", tuple(W_DTYPES))
def test_zero_rows_launch_nothing(hidden: int, w: str) -> None:
    _skip_unless_route(hidden, w_dtype=w)
    x = torch.empty(0, hidden, dtype=torch.bfloat16, device=_device())
    weight = torch.ones(hidden, dtype=W_DTYPES[w], device=_device())
    y, rstd = forward(x, weight, EPS[hidden])
    dw_buffer = torch.full((hidden,), float("nan"), device=_device())
    dx, dw = backward(x, x, weight, rstd, dw=dw_buffer)
    assert _LAUNCHED == []
    assert y.shape == (0, hidden) and rstd.shape == (0,) and dx.shape == (0, hidden)
    assert dw is dw_buffer and torch.equal(dw, torch.zeros_like(dw))


# --------------------------------------------------------------------------- repeats and the det package


def _det_records(hidden: int, residual: bool, w: str, rows: int) -> Optional[list[str]]:
    """The deterministic package's backward records for the call, or ``None`` when
    it has no route for it (it ships BF16 weights only)."""
    arch = det_loader.arch_for_capability(_capability())
    if w != "bf16" or arch is None:
        return None
    if det_loader.route_key(hidden, "bwd", residual) not in det_loader.ROUTES.get(
        arch, {}
    ):
        return None
    return list(det_loader.selected_modules(arch, hidden, "bwd", residual, rows=rows))


@functools.cache
def _kernel_text(build, library: str, prefix: str) -> str:
    """A library's kernel source with its package prefix replaced by a marker."""
    path = Path(build.__file__).parent / "csrc" / f"{library}_kernel.cu"
    return path.read_text().replace(prefix, "@prefix@")


def _program_id(name: str) -> str:
    """The program part of a module name (the digits after the package prefix,
    without the ``__<above>_<at_most>`` window suffix of a record that splits a
    det window): a det pair window of this package renders the det program of
    that name."""
    return name.split("__", 1)[0].rsplit("_", 1)[1]


# Single-kernel libraries whose ``dx`` reproduces a det program's ``dx`` bits
# (``dx`` has no cross-row reduction; 0 differing elements on B200 at the token
# counts named): library -> the det module whose bits it
# claims.  The claim holds where the det package selects that module; elsewhere
# the single is compared only.
DX_BITWISE_WITH = {
    # sm_100a H512 row (1280, inf); det module at T 16385
    "cake_rmsnorm_train_d0bde38aa443d42a5f04": "cake_rmsnorm_det_train_ad1a5a3c5f88e3d10036",
    # sm_100a H1536 row (36864, inf); det module at T 65536
    "cake_rmsnorm_train_ae610afdf77ff6714a5b": "cake_rmsnorm_det_train_b457c4d134e44476dc6f",
    # sm_100a H2048 column (0, 336]; det module at T 257
    "cake_rmsnorm_train_64e6aa82bb2c6d4ea978": "cake_rmsnorm_det_train_caf6978ed0a17fbf65b9",
    # sm_100a H2048 help row (7168, 11264] and (13608, 45056]; det module at T 16385
    "cake_rmsnorm_train_0fa4f255caf709d09f66": "cake_rmsnorm_det_train_ceeef043ac64f37fc529",
    # sm_100a H4096 column (0, 1152]; det module at T 257
    "cake_rmsnorm_train_4822793ea0c954c11af6": "cake_rmsnorm_det_train_59270fb249dcba1ff679",
    # sm_100a H4096 help row (4608, 36864]; det module at T 16385
    "cake_rmsnorm_train_15e157d2ee9080d206a9": "cake_rmsnorm_det_train_35a0b0ca6b2060313255",
    # sm_100a H6144 column (0, 1152]; det module at T 257 (the help row differs)
    "cake_rmsnorm_train_44508df7dab2ab688f85": "cake_rmsnorm_det_train_5c804aed7ad14b99e1d0",
    # sm_100a H6144 residual column (0, 1408]; the det residual program (every T)
    "cake_rmsnorm_train_864e84f3d874f850c091": "cake_rmsnorm_det_train_0dcdc563be377247044e",
    # sm_100a H6144 residual help row (1408, inf); the det residual program (every T)
    "cake_rmsnorm_train_3fbfdd750cbeb47032b5": "cake_rmsnorm_det_train_0dcdc563be377247044e",
}


def test_dx_bitwise_claims_name_shipped_programs() -> None:
    """Every :data:`DX_BITWISE_WITH` entry names a single-kernel library this
    package ships and a det module the det package ships (a renamed program
    fails here instead of silently dropping its claim)."""
    for library, det_name in DX_BITWISE_WITH.items():
        records = [r for r in loader.MODULES.values() if r["library"] == library]
        assert records, f"{library} is not a library of this package"
        assert all(
            (r["workspace_rule"] or {}).get("kind") == "claim_sum_f64" for r in records
        ), f"{library} is not a single-kernel library"
        assert det_name in det_loader.MODULES, f"{det_name} is not a det module"


def _dx_bitwise_claimed(
    records: list[str], det_records: list[str]
) -> tuple[bool, bool]:
    """``(claimed, same_source)`` for the window's main program against the det
    package's selected main program.  Claimed: the window runs the det program
    of that name (same program id: same tile and row mapping, and ``dx`` has no
    cross-row reduction), or a single kernel whose record names that det program
    (:data:`DX_BITWISE_WITH`).  ``same_source``: the two kernel sources are equal once
    the package prefixes are replaced (reported only: the det package and this
    package may render the same program from different source revisions)."""
    main = next(n for n in records if loader.MODULES[n]["role"] != "bwd_reduce")
    det_main = next(
        n for n in det_records if det_loader.MODULES[n]["role"] != "bwd_reduce"
    )
    same_source = _kernel_text(
        cake_build, loader.MODULES[main]["library"], "cake_rmsnorm_train_"
    ) == _kernel_text(
        det_build, det_loader.MODULES[det_main]["library"], "cake_rmsnorm_det_train_"
    )
    claim = DX_BITWISE_WITH.get(loader.MODULES[main]["library"])
    if claim is not None:
        return _program_id(claim) == _program_id(det_main), same_source
    return _program_id(main) == _program_id(det_main), same_source


# (label, claimed, differing dx elements) of every dx comparison against the det package in this process.
_DX_CLAIMS: list[tuple[str, bool, int]] = []


def _dx_vs_det(dx: torch.Tensor, dx_det: torch.Tensor) -> tuple[int, float, float]:
    """``(differing elements, max |dx - dx_det|, worst |dx - dx_det| / (|dx_det| + rms(dx_det)))``."""
    differing = int((dx.view(torch.int16) != dx_det.view(torch.int16)).sum().item())
    a, b = dx.double(), dx_det.double()
    if b.numel() == 0:
        return differing, 0.0, 0.0
    scale = b.pow(2).mean().sqrt().item()
    scale = scale if scale > 0.0 else 1.0
    return (
        differing,
        (a - b).abs().max().item(),
        ((a - b).abs() / (b.abs() + scale)).max().item(),
    )


def _repeat_params():
    for hidden in HIDDEN_SIZES:
        for rows in (257, 16385, 65536):
            for w in W_DTYPES:
                yield pytest.param(hidden, rows, False, w, id=f"h{hidden}-T{rows}-{w}")
    for rows in (7, 257, 16385, 65536):
        for w in W_DTYPES:
            yield pytest.param(6144, rows, True, w, id=f"res-h6144-T{rows}-{w}")


@pytest.mark.parametrize(("hidden", "rows", "residual", "w"), list(_repeat_params()))
def test_repeated_launches_against_fp64_and_det(
    hidden: int, rows: int, residual: bool, w: str
) -> None:
    """20 launches on one reused workspace: ``dx`` bitwise identical every time,
    ``dw`` within the budget every time (worst case reported), the workspace at
    rest after every call; where the det package serves the call, ``dx``
    bitwise equal to its ``dx`` when the window claims it (else within one BF16
    ulp of max(|dx|, rms(dx)), differing count reported) and ``dw`` within the
    budget of its ``dw`` (max |non-det - det| reported)."""
    _skip_unless_route(hidden, residual, w)
    case = _check(hidden, rows, residual=residual, w_dtype=W_DTYPES[w], key=11)
    det_records = _det_records(hidden, residual, w, rows)
    dx_det = dw_det = None
    if det_records is not None:
        dx_det, dw_det = det.cake_rmsnorm_det_train_backward(
            case.g, case.normalized, case.w, case.rstd, g_residual=case.g_h
        )
    ws = workspace_for(rows, hidden, _device(), residual=residual)
    worst, worst_tol, spread, dw_vs_det = 0.0, 0.0, 0.0, 0.0
    label = f"repeat h={hidden} T={rows} res={residual} w={w}"
    for _ in range(REPEATS):
        (dx, dw), records = _run_backward(
            case.g,
            case.normalized,
            case.w,
            case.rstd,
            g_residual=case.g_h,
            workspace=ws,
        )
        assert records == case.records, f"{label}: launched {records}"
        assert torch.equal(dx, case.dx), f"{label}: dx changed between launches"
        error, tol = _assert_dw(
            dw, case.dw_ref, case.g, case.normalized, case.rstd, label, case.form
        )
        worst, worst_tol = max(worst, error), max(worst_tol, tol)
        spread = max(spread, (dw - case.dw).abs().max().item())
        assert _workspace_at_rest(ws), f"{label}: workspace not at rest"
        if dw_det is not None:
            dw_vs_det = max(dw_vs_det, (dw - dw_det).abs().max().item())
            assert _rel_l2(dw, dw_det) <= DW_REL_L2, f"{label}: dw vs det"
    det_note = "det=none"
    if dx_det is not None:
        claimed, same_source = _dx_bitwise_claimed(case.records, det_records)
        differing, max_dx, worst_rel = _dx_vs_det(case.dx, dx_det)
        _DX_CLAIMS.append((label, claimed, differing))
        det_note = (
            f"dx_bitwise_claimed={claimed} same_source={same_source} "
            f"dx_differing={differing} max_dx_vs_det={max_dx:.3e} "
            f"max_dw_vs_det={dw_vs_det:.3e}"
        )
        if claimed:
            assert differing == 0, f"{label}: {differing} dx elements differ from det"
        assert worst_rel <= 2.0**-7, (
            f"{label}: dx differs from the det dx by {worst_rel:.3e} of "
            "max(|dx|, rms(dx)), above one BF16 ulp"
        )
    print(
        f"DW_WORST h={hidden} T={rows} res={residual} w={w} form={case.form} "
        f"repeats={REPEATS} rel_l2={worst:.3e} worst_tol_ratio={worst_tol:.3f} "
        f"max_abs_spread={spread:.3e} {det_note}"
    )


# H5120: the backward runs the plain det pair up to H5120_PLAIN_EDGE, the
# deterministic package's compensated det pair (compensated per-chunk dw sums)
# from there up to H5120_ACCURACY_EDGE, and the FP64-accumulating single kernel
# (on sm_103a its twosum form) above it.  The plain edge is the deterministic
# package's own switch to its compensated pair; the accuracy edge, per weight
# dtype, is the largest token count at which the compensated pair's dw is
# measured within ACCURACY_MARGIN of the per-element budget: T 65536, and T 57344
# for the FP32 weight on R200 (0.877 of the budget at T 65536).  The margin is
# taken against the FP64 reference at the kernel's FP32 rstd; against the FP64
# reference at the FP64 rstd (which differs from it by the forward's FP32 rstd
# rounding, the same for the deterministic package) the dw stays within the
# budget.  Worst dw element over its budget at the kernel's rstd, 60 inputs of
# each of three input families, BF16 and FP32 weight: B200 plain pair <= 0.236
# up to T 6144, compensated pair <= 0.781 up to T 65536, single <= 0.643 at
# T 65537; GB300 plain pair <= 0.245 up to T 5120, compensated pair <= 0.657 up
# to T 65536, twosum single <= 0.0006 at T 65537 and 81920; R200 plain pair
# <= 0.499 (216 SMs) and <= 0.545 (212 SMs) up to T 19968, compensated pair
# <= 0.767 (BF16 weight, up to T 65536; 0.837 at T 65536 on 216 SMs) and
# <= 0.732 (FP32 weight, up to T 57344), single <= 0.639 at T 65537 (BF16
# weight) and <= 0.633 at T 57345 (FP32 weight) (212 SMs).
H5120_PLAIN_EDGE = {"sm_100a": 6144, "sm_103a": 5120, "sm_107a": 19968}
H5120_ACCURACY_EDGE = {
    "sm_100a": {"bf16": 65536, "f32": 65536},
    "sm_103a": {"bf16": 65536, "f32": 65536},
    "sm_107a": {"bf16": 65536, "f32": 57344},
}
ACCURACY_MARGIN = 0.85
ACCURACY_SEEDS = 20


def _h5120_inputs(family: str, rows: int, w_dtype, k: int):
    """``(x, g, w)`` of input ``k``: ``test`` = this module's generator, ``plain``
    = one device generator seeded ``SEED + 7919 k`` drawing x, g ~ N(0, 1) and
    w = 1 + 0.1 N(0, 1) in that order."""
    if family == "test":
        x, _u, g, _g_h, w = _case_inputs(5120, rows, w_dtype=w_dtype, key=k)
        return x, g, w
    generator = torch.Generator(device=_device()).manual_seed(SEED + 7919 * k)

    def normal(*shape):
        return torch.randn(
            *shape, dtype=torch.float32, device=_device(), generator=generator
        )

    x = normal(rows, 5120).to(torch.bfloat16)
    g = normal(rows, 5120).to(torch.bfloat16)
    return x, g, (1.0 + 0.1 * normal(5120)).to(w_dtype)


@pytest.mark.parametrize("family", ("test", "plain"))
@pytest.mark.parametrize("w", tuple(W_DTYPES))
@pytest.mark.parametrize("side", ("plain", "compensated", "edge", "above"))
def test_h5120_dw_accuracy_edge(side: str, w: str, family: str) -> None:
    """At the H5120 windows -- the plain det pair's top (``H5120_PLAIN_EDGE``),
    the compensated det pair's first and last token counts, one above the
    accuracy edge (the single kernel) -- over ``ACCURACY_SEEDS`` inputs of each
    family: against the FP64 reference at the kernel's FP32 ``rstd``, a det
    pair's worst dw element stays within ``ACCURACY_MARGIN`` of its budget and
    the single kernel's within the budget; against the FP64 reference at the
    FP64 ``rstd`` every form's stays within the budget."""
    _skip_unless_route(5120, w_dtype=w)
    plain_edge, edge = H5120_PLAIN_EDGE[_arch()], H5120_ACCURACY_EDGE[_arch()][w]
    rows = {
        "plain": plain_edge,
        "compensated": plain_edge + 1,
        "edge": edge,
        "above": edge + 1,
    }[side]
    expected = _window_records(5120, "bwd", False, w, rows)
    form = _form(expected)
    assert form == ("single" if side == "above" else "pair"), (
        f"T={rows} runs {form}: the accuracy edge of this architecture is {edge}"
    )
    # the plain and the compensated det pair are different programs; the
    # compensated pair runs from one above the plain edge to the accuracy edge
    plain = _window_records(5120, "bwd", False, w, plain_edge)
    compensated = _window_records(5120, "bwd", False, w, plain_edge + 1)
    library = loader.MODULES[plain[0]]["library"]
    assert library != loader.MODULES[compensated[0]]["library"], (
        f"T={plain_edge} and T={plain_edge + 1} run the same program {library}"
    )
    assert compensated == _window_records(5120, "bwd", False, w, edge)
    eps = EPS[5120]
    worst, worst_fp64_rstd = 0.0, 0.0

    def over_budget(dw, ref):
        return float(
            ((dw.double() - ref).abs() / (DW_ATOL + DW_RTOL * ref.abs())).max()
        )

    for k in range(ACCURACY_SEEDS):
        x, g, weight = _h5120_inputs(family, rows, W_DTYPES[w], k)
        rstd = forward(x, weight, eps)[1]
        (_dx, dw), records = _run_backward(g, x, weight, rstd)
        assert records == expected, f"T={rows} launched {records}"
        xd = x.double()
        rstd_fp64 = torch.rsqrt(xd.pow(2).mean(-1, keepdim=True) + eps)
        worst = max(worst, over_budget(dw, _reference_backward(g, x, weight, rstd)[1]))
        worst_fp64_rstd = max(
            worst_fp64_rstd, over_budget(dw, (g.double() * xd * rstd_fp64).sum(0))
        )
    bound = ACCURACY_MARGIN if form == "pair" else 1.0
    print(
        f"H5120_EDGE arch={_arch()} edge={edge} T={rows} w={w} family={family} "
        f"form={form} seeds={ACCURACY_SEEDS} worst_error_over_budget={worst:.3f} "
        f"fp64_rstd={worst_fp64_rstd:.3f}"
    )
    assert worst <= bound, (
        f"H5120 T={rows} w={w} {family} inputs ({form}): worst dw element "
        f"{worst:.3f} of its budget > {bound}"
    )
    assert worst_fp64_rstd <= 1.0, (
        f"H5120 T={rows} w={w} {family} inputs ({form}): worst dw element "
        f"{worst_fp64_rstd:.3f} of its budget against the FP64-rstd reference > 1"
    )


@pytest.mark.parametrize("w", tuple(W_DTYPES))
@pytest.mark.parametrize("view", ("query", "key"))
def test_per_head_256_views(view: str, w: str) -> None:
    """The per-head shapes: a [T, 24, 256] query view of a [T, 24, 512] buffer
    (strides (12288, 512, 1)) and a contiguous [T, 4, 256] key view share one
    w[256]; dw sums over tokens and heads, rstd is [T, heads].  20 launches:
    ``dx`` bitwise stable and equal to the 2-D call on the collapsed rows, ``dw``
    within the budget every time (worst case reported)."""
    _skip_unless_route(256, w_dtype=w)
    rows = 1000
    heads = 24 if view == "query" else 4
    generator = _generator(256, rows, heads, w == "f32")
    if view == "query":
        buffer = torch.randn(rows, 24, 512, device=_device(), generator=generator).to(
            torch.bfloat16
        )
        x = buffer[:, :, :256]
        assert x.stride() == (12288, 512, 1)
    else:
        x = torch.randn(rows, 4, 256, device=_device(), generator=generator).to(
            torch.bfloat16
        )
    w_tensor = _weight(256, generator, W_DTYPES[w])
    g = torch.randn(x.shape, device=_device(), generator=generator).to(torch.bfloat16)
    y, rstd = forward(x, w_tensor, EPS[256])
    assert y.shape == x.shape and y.is_contiguous() and rstd.shape == x.shape[:-1]
    y_ref, rstd_ref, _n = _reference_forward(x, w_tensor, EPS[256])
    _assert_bf16_close(y, y_ref, f"y {view}")
    assert _rel_l2(rstd, rstd_ref) <= RSTD_REL_L2
    dx_ref, dw_ref = _reference_backward(g, x, w_tensor, rstd)
    expected = _window_records(256, "bwd", False, w, rows * heads)
    form = _form(expected)
    ws = workspace_for(rows * heads, 256, _device())
    dx0, worst, spread = None, 0.0, 0.0
    dw0 = None
    for _ in range(REPEATS):
        (dx, dw), records = _run_backward(g, x, w_tensor, rstd, workspace=ws)
        assert records == expected, records
        assert dx.shape == x.shape and dx.is_contiguous() and dw.shape == (256,)
        if dx0 is None:
            dx0, dw0 = dx, dw
            _assert_bf16_close(dx, dx_ref, f"dx {view}")
        assert torch.equal(dx, dx0)
        error, _tol = _assert_dw(dw, dw_ref, g, x, rstd, f"dw {view} w={w}", form)
        worst = max(worst, error)
        spread = max(spread, (dw - dw0).abs().max().item())
        assert _workspace_at_rest(ws)
    dx2, _dw2 = backward(
        g.reshape(-1, 256), x.reshape(-1, 256), w_tensor, rstd.reshape(-1)
    )
    assert torch.equal(dx0.reshape(-1, 256), dx2)
    print(
        f"DW_WORST per-head view={view} w={w} rows={rows * heads} form={form} "
        f"repeats={REPEATS} rel_l2={worst:.3e} max_abs_spread={spread:.3e} det=none"
    )


# --------------------------------------------------------------------------- outputs, workspace, streams, graphs


@pytest.mark.parametrize("hidden", (512, 4096))
@pytest.mark.parametrize("w", tuple(W_DTYPES))
def test_garbage_filled_dw_is_overwritten(hidden: int, w: str) -> None:
    _skip_unless_route(hidden, w_dtype=w)
    case = _check(hidden, 1000, w_dtype=W_DTYPES[w])
    for fill in (float("nan"), 1e30, -7.0):
        dw = torch.full((hidden,), fill, device=_device())
        _dx2, dw_out = backward(case.g, case.x, case.w, case.rstd, dw=dw)
        assert dw_out is dw and torch.isfinite(dw).all()
        _assert_dw(
            dw, case.dw_ref, case.g, case.x, case.rstd, f"dw fill={fill}", case.form
        )


def _reuse_call(ws, hidden, residual, w, rows, key):
    """One forward and backward on the caller's workspace ``ws``, its pair
    partial rows poisoned (NaN) first: the call rewrites every row it reads."""
    eps = EPS[hidden]
    ws[loader.PAIR_ROWS_OFFSET :].fill_(0xFF)
    x, u, g, g_h, weight = _case_inputs(
        hidden, rows, residual=residual, w_dtype=W_DTYPES[w], key=key
    )
    outputs = forward(x, weight, eps, residual=u)
    normalized = outputs[2] if residual else x
    rstd = outputs[1]
    (dx, dw), records = _run_backward(
        g, normalized, weight, rstd, g_residual=g_h, workspace=ws
    )
    expected = _window_records(hidden, "bwd", residual, w, rows)
    label = f"h={hidden} res={residual} w={w} T={rows}"
    assert records == expected, f"{label}: launched {records}"
    dx_ref, dw_ref = _reference_backward(g, normalized, weight, rstd, g_h=g_h)
    _assert_bf16_close(dx, dx_ref, f"dx {label}")
    _assert_dw(dw, dw_ref, g, normalized, rstd, f"dw {label}", _form(expected))
    return label, _form(expected)


@pytest.mark.parametrize("hidden", (512, 2048, 6144))
def test_workspace_reused_across_token_counts_and_weight_dtypes(hidden: int) -> None:
    """One buffer serves a width's calls of every token count up to its rows,
    every window and both weight dtypes with no host write but NaN poison in its
    pair partial rows before each call; at rest after every call.  (Reuse across
    widths, residual flags and stale bytes: ``test_cake_rmsnorm_train_api.py``.)"""
    _skip_unless_route(hidden)
    ws = torch.zeros(_bytes(65536, hidden), dtype=torch.uint8, device=_device())
    w_dtypes = ["bf16"] + (["f32"] if _routes(hidden, "bwd", False, "f32") else [])
    forms = []
    for index, rows in enumerate((7, 65536, 1, 257, 16385, 4096, 2)):
        w = w_dtypes[index % len(w_dtypes)]
        _label, form = _reuse_call(ws, hidden, False, w, rows, key=29)
        assert _workspace_at_rest(ws), f"workspace state after h={hidden} T={rows}"
        forms.append(f"{rows}:{w}:{form}")
    print(f"REUSE h={hidden} " + " ".join(forms))


@pytest.mark.parametrize(
    ("hidden", "residual", "w"),
    [
        pytest.param(4096, False, "bf16", id="h4096-bf16"),
        pytest.param(6144, True, "f32", id="res-h6144-f32"),
    ],
)
def test_two_streams_with_their_own_workspaces(hidden: int, residual: bool, w: str):
    _skip_unless_route(hidden, residual, w)
    cases = []
    for rows in (16385, 65536):
        x, u, g, g_h, weight = _case_inputs(
            hidden, rows, residual=residual, w_dtype=W_DTYPES[w], key=37
        )
        outputs = forward(x, weight, EPS[hidden], residual=u)
        normalized, rstd = (outputs[2] if residual else x), outputs[1]
        dx_ref, dw_ref = _reference_backward(g, normalized, weight, rstd, g_h=g_h)
        form = _form(_window_records(hidden, "bwd", residual, w, rows))
        cases.append((g, g_h, normalized, weight, rstd, dx_ref, dw_ref, form))
    streams = [torch.cuda.Stream() for _ in cases]
    workspaces = [
        workspace_for(65536, hidden, _device(), residual=residual) for _ in cases
    ]
    torch.cuda.synchronize()
    results = []
    for _round in range(5):
        for stream, ws, case in zip(streams, workspaces, cases, strict=True):
            g, g_h, normalized, weight, rstd = case[:5]
            with torch.cuda.stream(stream):
                results.append(
                    backward(g, normalized, weight, rstd, g_residual=g_h, workspace=ws)
                )
    torch.cuda.synchronize()
    for index, (dx, dw) in enumerate(results):
        g, _g_h, normalized, _weight, rstd, dx_ref, dw_ref, form = cases[
            index % len(cases)
        ]
        _assert_bf16_close(dx, dx_ref, f"stream dx {index}")
        _assert_dw(dw, dw_ref, g, normalized, rstd, f"stream dw {index}", form)
    assert all(_workspace_at_rest(ws) for ws in workspaces)


def _graph_params():
    for hidden, residual, w in (
        (512, False, "bf16"),
        (6144, False, "bf16"),
        (6144, True, "bf16"),
        (6144, True, "f32"),
        (4096, False, "f32"),
        (5120, False, "f32"),
        (256, False, "bf16"),
    ):
        for rows in (64, 4096):
            prefix = "res-" if residual else ""
            yield pytest.param(
                hidden, residual, w, rows, id=f"{prefix}h{hidden}-{w}-T{rows}"
            )


@pytest.mark.parametrize(("hidden", "residual", "w", "rows"), list(_graph_params()))
def test_cuda_graph_capture_and_replay(hidden: int, residual: bool, w: str, rows: int):
    _skip_unless_route(hidden, residual, w)
    x, u, g, g_h, weight = _case_inputs(
        hidden, rows, residual=residual, w_dtype=W_DTYPES[w], key=41
    )
    eps = EPS[hidden]
    ws = workspace_for(rows, hidden, _device(), residual=residual)
    outputs = forward(x, weight, eps, residual=u)  # warm the launch path
    normalized, rstd = (outputs[2] if residual else x), outputs[1]
    dx, dw = backward(g, normalized, weight, rstd, g_residual=g_h, workspace=ws)
    out_y, out_rstd = torch.empty_like(outputs[0]), torch.empty_like(rstd)
    out_h = torch.empty_like(normalized) if residual else None
    out_dx, out_dw = torch.empty_like(dx), torch.full_like(dw, float("nan"))
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        forward(
            x, weight, eps, residual=u, out=out_y, rstd=out_rstd, residual_out=out_h
        )
        backward(
            g,
            out_h if residual else x,
            weight,
            out_rstd,
            g_residual=g_h,
            workspace=ws,
            dx=out_dx,
            dw=out_dw,
        )
    dx_ref, dw_ref = _reference_backward(g, normalized, weight, rstd, g_h=g_h)
    form = _form(_window_records(hidden, "bwd", residual, w, rows))
    for replay in range(3):
        out_dw.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out_y, outputs[0]) and torch.equal(out_rstd, rstd)
        if residual:
            assert torch.equal(out_h, normalized)
        assert torch.equal(out_dx, dx)
        _assert_dw(out_dw, dw_ref, g, normalized, rstd, f"graph replay {replay}", form)
        assert _workspace_at_rest(ws)


# --------------------------------------------------------------------------- autograd


def _autograd_params():
    for hidden in (512, 4096):
        for w in W_DTYPES:
            yield pytest.param(hidden, False, w, id=f"h{hidden}-{w}")
    for w in W_DTYPES:
        yield pytest.param(6144, True, w, id=f"res-h6144-{w}")


def _assert_autograd_dw(grad_w, weight, dw_ref, g, normalized, rstd, label, form):
    """The autograd weight gradient is the kernel's FP32 ``dw`` (the last
    backward buffer this package wrote) at the det budget, cast to the weight's
    dtype bit for bit."""
    assert _DW_BUFFERS, f"{label}: no backward module of this package ran"
    dw32 = _DW_BUFFERS[-1].clone()
    assert grad_w.dtype == weight.dtype, label
    _assert_dw(dw32, dw_ref, g, normalized, rstd, label, form)
    assert torch.equal(grad_w, dw32.to(weight.dtype)), (
        f"{label}: the autograd dw is not the kernel's dw in {weight.dtype}"
    )
    return dw32


@pytest.mark.parametrize(("hidden", "residual", "w"), list(_autograd_params()))
def test_autograd_matches_fp64(hidden: int, residual: bool, w: str) -> None:
    _skip_unless_route(hidden, residual, w)
    rows = 1000
    x, u, g, g_h, weight = _case_inputs(
        hidden, rows, residual=residual, w_dtype=W_DTYPES[w], key=43
    )
    xa = x.clone().requires_grad_(True)
    wa = weight.clone().requires_grad_(True)
    ua = u.clone().requires_grad_(True) if residual else None
    outputs = rmsnorm(xa, wa, EPS[hidden], residual=ua)
    if residual:
        y, h_new = outputs
        torch.autograd.backward((y, h_new), (g, g_h))
    else:
        y = outputs
        y.backward(g)
    y_ref, _rstd_ref, _normalized_ref = _reference_forward(
        x, weight, EPS[hidden], residual=u
    )
    _assert_bf16_close(y.detach(), y_ref, "autograd y")
    # the backward reference at the FP32 rstd the forward saved (same program, same bits)
    outputs_fwd = forward(x, weight, EPS[hidden], residual=u)
    normalized, rstd = (outputs_fwd[2] if residual else x), outputs_fwd[1]
    dx_ref, dw_ref = _reference_backward(g, normalized, weight, rstd, g_h=g_h)
    _assert_bf16_close(xa.grad, dx_ref, "autograd dx")
    if residual:
        assert torch.equal(ua.grad, xa.grad)
    form = _form(_window_records(hidden, "bwd", residual, w, rows))
    _assert_autograd_dw(
        wa.grad, weight, dw_ref, g, normalized, rstd, "autograd dw", form
    )


@pytest.mark.parametrize(("hidden", "residual", "w"), list(_autograd_params()))
def test_autograd_accumulates_into_existing_grads(hidden: int, residual: bool, w: str):
    """Gradients add into existing ``.grad`` tensors (two backwards through one
    retained graph, on top of preset gradients) exactly as autograd adds them;
    every kernel ``dw`` is at the det budget; the graph's saved tensors are
    released after the last backward, so a third one raises."""
    _skip_unless_route(hidden, residual, w)
    rows = 4096
    eps = EPS[hidden]
    x, u, g, g_h, weight = _case_inputs(
        hidden, rows, residual=residual, w_dtype=W_DTYPES[w], key=47
    )
    generator = _generator(hidden, rows, 53)
    gx0 = _rows(rows, hidden, generator=generator)
    gw0 = _weight(hidden, generator, W_DTYPES[w]) - 1.0
    xa = x.clone().requires_grad_(True)
    wa = weight.clone().requires_grad_(True)
    ua = u.clone().requires_grad_(True) if residual else None
    xa.grad, wa.grad = gx0.clone(), gw0.clone()
    if residual:
        ua.grad = gx0.clone()
    outputs = rmsnorm(xa, wa, eps, residual=ua)
    tensors = outputs if residual else (outputs,)
    grads = (g, g_h) if residual else (g,)
    outputs_fwd = forward(x, weight, eps, residual=u)
    normalized, rstd = (outputs_fwd[2] if residual else x), outputs_fwd[1]
    dx_k, _dw = backward(g, normalized, weight, rstd, g_residual=g_h)
    dx_ref, dw_ref = _reference_backward(g, normalized, weight, rstd, g_h=g_h)
    form = _form(_window_records(hidden, "bwd", residual, w, rows))
    torch.autograd.backward(tensors, grads, retain_graph=True)
    dw1 = _DW_BUFFERS[-1].clone()
    _assert_dw(dw1, dw_ref, g, normalized, rstd, "accumulate dw 1", form)
    expected_x = gx0 + dx_k
    expected_w = gw0 + dw1.to(weight.dtype)
    assert torch.equal(xa.grad, expected_x)
    assert torch.equal(wa.grad, expected_w)
    if residual:
        assert torch.equal(ua.grad, expected_x)
    torch.autograd.backward(tensors, grads)
    dw2 = _DW_BUFFERS[-1].clone()
    _assert_dw(dw2, dw_ref, g, normalized, rstd, "accumulate dw 2", form)
    assert torch.equal(xa.grad, expected_x + dx_k)
    assert torch.equal(wa.grad, expected_w + dw2.to(weight.dtype))
    if residual:
        assert torch.equal(ua.grad, expected_x + dx_k)
    with pytest.raises(RuntimeError, match="backward through the graph a second time"):
        torch.autograd.backward(tensors, grads)


@pytest.mark.parametrize(("hidden", "residual", "w"), list(_autograd_params()))
def test_autograd_saves_only_its_inputs_and_rstd(hidden: int, residual: bool, w: str):
    """The autograd function saves three tensors -- the normalized input (the
    caller's ``x`` or the returned ``h_new``, not a copy), the weight and the
    FP32 ``[T]`` rstd -- and holds nothing else: after the backward and the
    outputs are dropped, the device memory is the inputs and their gradients."""
    _skip_unless_route(hidden, residual, w)
    rows = 4096
    eps = EPS[hidden]
    x, u, g, g_h, weight = _case_inputs(
        hidden, rows, residual=residual, w_dtype=W_DTYPES[w], key=59
    )
    xa = x.clone().requires_grad_(True)
    wa = weight.clone().requires_grad_(True)
    ua = u.clone().requires_grad_(True) if residual else None
    saved: list[torch.Tensor] = []

    def pack(tensor):
        saved.append(tensor)
        return tensor

    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        outputs = rmsnorm(xa, wa, eps, residual=ua)
    y = outputs[0] if residual else outputs
    normalized = outputs[1] if residual else xa
    assert len(saved) == 3, [(tuple(t.shape), t.dtype) for t in saved]
    others = [
        t for t in saved if t.data_ptr() not in (normalized.data_ptr(), wa.data_ptr())
    ]
    assert len(others) == 1, [(tuple(t.shape), t.dtype) for t in saved]
    assert others[0].dtype == torch.float32 and tuple(others[0].shape) == (rows,)
    torch.cuda.synchronize()
    held = torch.cuda.memory_allocated() - base
    outputs_bytes = y.numel() * y.element_size() + (
        normalized.numel() * normalized.element_size() if residual else 0
    )
    # the outputs plus the FP32 rstd (allocator blocks of 512 bytes)
    assert held <= outputs_bytes + 4 * rows + 2048, (held, outputs_bytes)
    saved.clear()  # drop this test's references to the saved tensors
    others.clear()
    if residual:
        torch.autograd.backward(outputs, (g, g_h))
    else:
        outputs.backward(g)
    del outputs, y, normalized
    _DW_BUFFERS.clear()  # the launch recorder's references to the kernel's dw
    torch.cuda.synchronize()
    grads = [xa.grad, wa.grad] + ([ua.grad] if residual else [])
    storages = {
        t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for t in grads
    }
    grads_bytes = sum(storages.values())
    left = torch.cuda.memory_allocated() - base
    assert left <= grads_bytes + 2048, (
        f"{left} bytes held after the backward, gradients {grads_bytes}"
    )


@pytest.mark.parametrize(("hidden", "residual", "w"), list(_autograd_params()))
def test_checkpoint_replays_the_same_forward(hidden: int, residual: bool, w: str):
    """Non-reentrant activation checkpointing: the replayed forward gives the
    same ``y`` and ``dx`` bits as the plain autograd call; ``dw`` of both is at
    the det budget; two checkpointed micro-batches accumulate into one ``.grad``."""
    _skip_unless_route(hidden, residual, w)
    rows = 1000
    eps = EPS[hidden]
    x, u, g, g_h, weight = _case_inputs(
        hidden, rows, residual=residual, w_dtype=W_DTYPES[w], key=61
    )
    outputs_fwd = forward(x, weight, eps, residual=u)
    normalized, rstd = (outputs_fwd[2] if residual else x), outputs_fwd[1]
    dx_ref, dw_ref = _reference_backward(g, normalized, weight, rstd, g_h=g_h)
    form = _form(_window_records(hidden, "bwd", residual, w, rows))

    def step(xt, wt, ut):
        out = rmsnorm(xt, wt, eps, residual=ut)
        return out if residual else (out,)

    def run(checkpointed: bool, wt):
        xt = x.clone().requires_grad_(True)
        ut = u.clone().requires_grad_(True) if residual else None
        if checkpointed:
            outs = torch.utils.checkpoint.checkpoint(
                step, xt, wt, ut, use_reentrant=False
            )
        else:
            outs = step(xt, wt, ut)
        torch.autograd.backward(outs, (g, g_h) if residual else (g,))
        return outs[0].detach(), xt.grad

    wa = weight.clone().requires_grad_(True)
    y_ckpt, dx_ckpt = run(True, wa)
    dw_ckpt = _assert_autograd_dw(
        wa.grad, weight, dw_ref, g, normalized, rstd, "checkpoint dw", form
    )
    wb = weight.clone().requires_grad_(True)
    y_plain, dx_plain = run(False, wb)
    _assert_autograd_dw(wb.grad, weight, dw_ref, g, normalized, rstd, "plain dw", form)
    assert torch.equal(y_ckpt, y_plain) and torch.equal(dx_ckpt, dx_plain)
    _assert_bf16_close(dx_ckpt, dx_ref, "checkpoint dx")
    first = wa.grad.clone()
    run(True, wa)  # a second micro-batch accumulates
    dw_second = _DW_BUFFERS[-1].clone()
    _assert_dw(dw_second, dw_ref, g, normalized, rstd, "checkpoint dw 2", form)
    assert torch.equal(wa.grad, first + dw_second.to(weight.dtype))
    assert torch.equal(first, dw_ckpt.to(weight.dtype))


def _backward_launches(fn) -> list[str]:
    start = len(_LAUNCHED)
    fn()
    return [n for n in _LAUNCHED[start:] if loader.MODULES[n]["role"].startswith("bwd")]


@pytest.mark.parametrize("w", tuple(W_DTYPES))
def test_backward_resolves_the_mode_again_when_it_runs(w: str) -> None:
    """The autograd backward resolves ``deterministic=False`` again when it runs.
    Determinism switched on between forward and backward refuses it
    (``RuntimeError``, nothing launched).  ``warn_only`` switched on there warns
    and runs the det package's backward (BF16 weight: the det bits, no module of
    this package), or raises one ``RuntimeError`` naming what the det package
    lacks (FP32 weight).  A forward resolved under ``warn_only`` (BF16: warns
    and runs this package's forward programs; FP32: refused before any work)
    followed by a backward with determinism off runs this package's backward."""
    hidden, rows = 512, 257
    _skip_unless_route(hidden, w_dtype=w)
    eps = EPS[hidden]
    x, _u, g, _g_h, weight = _case_inputs(hidden, rows, w_dtype=W_DTYPES[w], key=67)
    expected = _window_records(hidden, "bwd", False, w, rows)
    rstd = forward(x, weight, eps)[1]

    def graph():
        xa = x.clone().requires_grad_(True)
        wa = weight.clone().requires_grad_(True)
        return xa, wa, rmsnorm(xa, wa, eps)

    # strict determinism at backward time
    _xa, _wa, y = graph()
    torch.use_deterministic_algorithms(True)
    start = len(_LAUNCHED)
    with pytest.raises(RuntimeError, match="deterministic=True"):
        y.backward(g)
    assert _LAUNCHED[start:] == [], "a refused backward launched a module"
    torch.use_deterministic_algorithms(False)

    # warn_only at backward time
    xa, wa, y = graph()
    torch.use_deterministic_algorithms(True, warn_only=True)
    start = len(_LAUNCHED)
    if w == "bf16":
        with pytest.warns(UserWarning, match="deterministic package's backward"):
            y.backward(g)
        assert _LAUNCHED[start:] == [], "the det fallback launched this package"
        dx_det, dw_det = det.cake_rmsnorm_det_train_backward(g, x, weight, rstd)
        assert torch.equal(xa.grad, dx_det)
        assert torch.equal(wa.grad, dw_det.to(wa.dtype))
    else:
        with pytest.raises(RuntimeError, match="only deterministic=False serves"):
            y.backward(g)
        assert _LAUNCHED[start:] == []
    torch.use_deterministic_algorithms(False)

    # warn_only at forward time, determinism off at backward time
    torch.use_deterministic_algorithms(True, warn_only=True)
    if w == "f32":
        start = len(_LAUNCHED)
        with pytest.raises(RuntimeError, match="only deterministic=False serves"):
            graph()
        assert _LAUNCHED[start:] == []
        torch.use_deterministic_algorithms(False)
        return
    with pytest.warns(UserWarning, match="this package's forward programs"):
        xa, wa, y = graph()
    torch.use_deterministic_algorithms(False)
    launched = _backward_launches(lambda: y.backward(g))
    assert launched == expected, launched
    dx_ref, dw_ref = _reference_backward(g, x, weight, rstd)
    _assert_bf16_close(xa.grad, dx_ref, "re-resolved dx")
    _assert_autograd_dw(
        wa.grad, weight, dw_ref, g, x, rstd, "re-resolved dw", _form(expected)
    )


# --------------------------------------------------------------------------- argument validation


def _refused(error: type, call, why: str) -> None:
    """``call`` raises ``error`` before launching anything."""
    start = len(_LAUNCHED)
    with pytest.raises(error):
        call()
    assert _LAUNCHED[start:] == [], f"refused call ({why}) launched {_LAUNCHED[start:]}"


@pytest.mark.parametrize("hidden", (512, 6144))
def test_argument_validation(hidden: int) -> None:
    """Wrong dtypes, shapes, devices, layouts, output buffers and workspaces
    (too small for the call, wrong dtype, on the CPU, misaligned, 2-D) are
    refused with ``ValueError`` before any launch (the ``deterministic`` flag's
    type: ``test_cake_rmsnorm_train_api.py``)."""
    _skip_unless_route(hidden)
    eps = EPS[hidden]
    rows = 4096
    x, _u, g, _g_h, weight = _case_inputs(hidden, rows, key=71)
    _y, rstd = forward(x, weight, eps)
    device = _device()
    refusals = (
        (lambda: forward(x.float(), weight, eps), "x dtype"),
        (lambda: forward(x, weight.half(), eps), "w dtype"),
        (lambda: forward(x[:, 1:], weight[1:], eps), "no route for hidden - 1"),
        (lambda: forward(x, weight[: hidden // 2], eps), "w shape"),
        (lambda: forward(x.cpu(), weight.cpu(), eps), "cpu"),
        (
            lambda: forward(
                x, weight, eps, out=torch.empty(rows, hidden, device=device)
            ),
            "out dtype",
        ),
        (
            lambda: forward(
                x.view(rows, 4, hidden // 4).transpose(1, 2), weight[:4], eps
            ),
            "non-collapsible N-D",
        ),
        (lambda: backward(g, x, weight, rstd.double()), "rstd dtype"),
        (lambda: backward(g, x, weight, rstd[:-1]), "rstd shape"),
        (lambda: backward(g[:-1], x, weight, rstd), "g shape"),
        (
            lambda: backward(
                g, x, weight, rstd, dx=torch.empty(rows, hidden, device=device)
            ),
            "dx dtype",
        ),
        (
            lambda: backward(
                g, x, weight, rstd, dw=torch.empty(hidden + 1, device=device)
            ),
            "dw shape",
        ),
        (
            lambda: pkg.cake_rmsnorm_train_backward_workspace_bytes(
                -1, hidden, device, deterministic=False
            ),
            "negative rows",
        ),
    )
    for call, why in refusals:
        _refused(ValueError, call, why)
    needed = _bytes(rows, hidden)
    workspaces = [
        (torch.zeros(256, dtype=torch.uint8, device=device), "tiny"),
        (
            torch.zeros(loader.COUNTERS_BYTES, dtype=torch.uint8, device=device),
            "counters only",
        ),
        (torch.zeros(needed // 4, dtype=torch.float32, device=device), "dtype"),
        (torch.zeros(needed, dtype=torch.uint8), "cpu"),
        (torch.zeros(needed + 512, dtype=torch.uint8, device=device)[1:], "unaligned"),
        (torch.zeros(2, needed, dtype=torch.uint8, device=device), "2-D"),
    ]
    if _form(_window_records(hidden, "bwd", False, "bf16", rows)) == "pair":
        # the documented size of a pair window's call is that call's own layout
        workspaces.append(
            (
                torch.zeros(needed - 256, dtype=torch.uint8, device=device),
                "one block short",
            )
        )
    for ws, why in workspaces:
        _refused(
            ValueError, lambda ws=ws: backward(g, x, weight, rstd, workspace=ws), why
        )


# --------------------------------------------------------------------------- build manifest (no device)


@pytest.mark.cpu_only
def test_every_library_carries_its_architecture_floor_and_flags() -> None:
    """Every ``build.json`` library is built for exactly its architecture's code
    (``-gencode=arch=compute_XXXa,code=sm_XXXa``), carries that architecture's
    CUDA floor (sm_100a 12.8, sm_103a 12.9, sm_107a 13.4) and the exact-math
    flags; the package floor is the lowest of them; every architecture ships
    every route (the feature x architecture matrix)."""
    manifest = cake_build.manifest()
    assert manifest["schema"] == "cake-build/1"
    assert set(manifest["arch_flags"]) == set(loader.ARCHES) == set(MIN_CUDA)
    for arch, flags in manifest["arch_flags"].items():
        assert flags == [GENCODE[arch]], (arch, flags)
    requires = manifest["common"]["requires"]["cuda_toolkit"]["min"]
    assert requires == min(
        MIN_CUDA.values(), key=lambda v: tuple(map(int, v.split(".")))
    )
    libraries = cake_build.libraries_by_name()
    per_arch = {arch: 0 for arch in loader.ARCHES}
    for name, entry in libraries.items():
        arch = entry["arch"]
        assert name.endswith(f"_{arch}"), name
        assert entry["min_cuda"] == MIN_CUDA[arch], (name, entry["min_cuda"])
        assert entry["gencode"] == GENCODE[arch], (name, entry["gencode"])
        flags = entry["nvcc_flags"]
        assert all(flag in flags for flag in ACCURACY_FLAGS), (name, flags)
        assert "--use_fast_math" not in flags, name
        per_arch[arch] += 1
    referenced = {
        loader.library_name(name, arch)
        for name, record in loader.MODULES.items()
        for arch in record["arches"]
    }
    assert referenced == set(libraries)
    routes = {arch: set(loader.ROUTES[arch]) for arch in loader.ARCHES}
    first = routes[loader.ARCHES[0]]
    assert all(keys == first for keys in routes.values()), {
        arch: sorted(keys ^ first) for arch, keys in routes.items()
    }
    for key in sorted(first):
        kind, hidden = key.split("_", 1)[0], int(key.rsplit("_h", 1)[1])
        assert hidden in loader.HIDDEN_SIZES and kind in loader.KINDS, key
    print(
        "MANIFEST "
        + " ".join(
            f"{arch}:{MIN_CUDA[arch]}:{per_arch[arch]}libs" for arch in loader.ARCHES
        )
        + f" routes={len(first)}"
    )


@pytest.mark.cpu_only
def test_registry_records_run_on_the_exported_architectures() -> None:
    for arch, by_route in loader.ROUTES.items():
        assert arch in loader.ARCHES
        for route, names in by_route.items():
            for name in names:
                record = loader.MODULES[name]
                assert arch in record["arches"], (arch, route, name)
                assert loader.library_name(name, arch) in cake_build.libraries_by_name()


# --------------------------------------------------------------------------- aggregate (runs last)


def test_dw_not_worse_than_eager_fp32_over_the_matrix() -> None:
    """Median readings of every dw case this process checked, per backward form:
    a single-kernel window's dw error is not worse than the eager FP32 reduction
    (also gated per case), and the det pair windows keep the det package's
    median gate.  Skipped when the module's dw cases did not run in this process
    (``-k`` selection)."""
    if len(_DW_ERRORS) < 8:
        pytest.skip("needs the dw cases of this module in the same process")
    counts = {form: 0 for form in ("single", "pair")}
    for form in counts:
        cases = [item for item in _DW_ERRORS if item[1] == form]
        counts[form] = len(cases)
        if not cases:
            continue
        ratios = [
            (k / e) if e > 0 else (0.0 if k == 0 else float("inf"))
            for _label, _form_name, k, e in cases
        ]
        median = statistics.median(ratios)
        worst = max(cases, key=lambda item: item[2])
        worst_ratio = max(ratios)
        print(
            f"DW_VS_EAGER form={form} cases={len(cases)} median_ratio={median:.3f} "
            f"worst_ratio={worst_ratio:.3f} worst_rel_l2={worst[2]:.3e} at {worst[0]}"
        )
        assert median <= 1.0, (
            f"{form}: median dw error ratio kernel/eager-FP32 {median:.3f} > 1 over "
            f"{len(cases)} cases (worst kernel rel-L2 {worst[2]:.3e} at {worst[0]})"
        )
    print(f"FORMS single={counts['single']} pair={counts['pair']}")
    claimed = [item for item in _DX_CLAIMS if item[1]]
    print(
        f"DX_CLAIMS compared={len(_DX_CLAIMS)} claimed={len(claimed)} "
        f"unclaimed_differing={sum(1 for item in _DX_CLAIMS if not item[1] and item[2])}"
    )
    # A det comparison that never claims bitwise dx would make that gate vacuous.
    assert not _DX_CLAIMS or claimed, "no window claims the det dx bits"
